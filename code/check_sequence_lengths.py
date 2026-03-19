#!/usr/bin/env python3
"""Check actual sequence lengths in the corpus to determine optimal max_seq_length."""

import argparse
import json
import numpy as np
from .arithmetic_tokenizer import ArithmeticBPETokenizer
import heapq

def analyze_corpus_lengths(corpus_path, tokenizer_path, max_samples=None, corpus_type='foundational'):
    """Analyze sequence lengths in a corpus.
    
    Args:
        corpus_path: Path to corpus file
        tokenizer_path: Path to tokenizer directory
        max_samples: Maximum number of samples to analyze (None = all)
        corpus_type: Type of corpus ('foundational' or 'instruction')
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"\nAnalyzing corpus: {corpus_path}")
    print(f"Corpus type: {corpus_type}")
    if max_samples:
        print(f"Sampling: first {max_samples} lines")
    else:
        print("Analyzing: all lines")
    
    lengths = []
    top_k = 5
    longest_heap = []
    skipped = 0
    # Track longest problem
    max_problem_len = 0
    max_problem_text = ""
    max_problem_idx = -1
    longest_problems = []
    problem_lengths = []
    problem_skipped = 0
    with open(corpus_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                # --- Measure problem length only ---
                # --- problem-only length ---
                try:
                    if 'problem' in entry and isinstance(entry['problem'], str) and entry['problem'].strip():
                        p_tokens = tokenizer.encode(entry['problem'], add_special_tokens=True)
                        problem_lengths.append(len(p_tokens))
                    else:
                        problem_skipped += 1
                except Exception:
                    problem_skipped += 1
                if 'problem' in entry:
                    problem_text = entry['problem']
                    problem_tokens = tokenizer.encode(problem_text, add_special_tokens=True)
                    problem_len = len(problem_tokens)

                    if problem_len > max_problem_len:
                        max_problem_len = problem_len
                        max_problem_text = problem_text
                        max_problem_idx = i
                # Handle different corpus formats
                # ---- Measure problem only ----
                if 'problem' in entry:
                    problem_text = entry['problem']
                    problem_tokens = tokenizer.encode(problem_text, add_special_tokens=True)
                    problem_len = len(problem_tokens)

                    item = (problem_len, i, problem_text)

                    if len(longest_problems) < top_k:
                        heapq.heappush(longest_problems, item)
                    else:
                        if problem_len > longest_problems[0][0]:
                            heapq.heapreplace(longest_problems, item)
                if corpus_type == 'foundational':
                    text = entry['problem'] + ' ' + entry['solution']
                elif corpus_type == 'instruction':
                    # Instruction format: problem + "<think>" prompt + solution
                    if 'problem' in entry and 'solution' in entry:
                        prompt = entry['problem'] + ' <think>'
                        text = prompt + ' ' + entry['solution']
                    else:
                        # Fallback for alternative schemas
                        text = entry.get('prompt', '') + ' ' + entry.get('response', '')
                else:
                    # Generic: concatenate all text fields
                    text = ' '.join(str(v) for v in entry.values() if isinstance(v, str))
                
                tokens = tokenizer.encode(text, add_special_tokens=True)
                lengths.append(len(tokens))
                L = len(tokens)

                # Maintain a min-heap of top_k longest sequences
                item = (L, i, text, tokens)
                if len(longest_heap) < top_k:
                    heapq.heappush(longest_heap, item)
                else:
                    if L > longest_heap[0][0]:
                        heapq.heapreplace(longest_heap, item)
                
            except Exception:
                skipped += 1
                continue
    
    if not lengths:
        print("\nERROR: No valid sequences found!")
        return
    
    # Calculate statistics
    lengths_array = np.array(lengths)
    percentiles = [50, 75, 90, 95, 99, 99.5, 100]
    
    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH ANALYSIS")
    print("=" * 60)
    print(f"\nTotal sequences analyzed: {len(lengths)}")
    if skipped > 0:
        print(f"Skipped (parse errors): {skipped}")
    
    print("\nBasic Statistics:")
    print(f"  Min length:     {np.min(lengths_array):>6} tokens")
    print(f"  Max length:     {np.max(lengths_array):>6} tokens")
    print(f"  Mean length:    {np.mean(lengths_array):>6.1f} tokens")
    print(f"  Median length:  {np.median(lengths_array):>6.1f} tokens")
    print(f"  Std deviation:  {np.std(lengths_array):>6.1f} tokens")
    
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(lengths_array, p)
        print(f"  {p:>5.1f}th percentile: {val:>6.0f} tokens")
    
    print("\nCoverage by max_seq_length:")
    thresholds = [64, 128, 192, 256, 384, 512, 768, 1024]
    for threshold in thresholds:
        count = np.sum(lengths_array <= threshold)
        pct = count / len(lengths_array) * 100
        truncated = len(lengths_array) - count
        print(f"  max_seq_length={threshold:>4}: {count:>6}/{len(lengths_array)} ({pct:>5.1f}%) | {truncated:>5} truncated")
    
    print("\nRecommendations:")
    # Find threshold that covers 95% and 99%
    p95 = np.percentile(lengths_array, 95)
    p99 = np.percentile(lengths_array, 99)
    
    print(f"  • For 95% coverage: max_seq_length >= {int(np.ceil(p95))}")
    print(f"  • For 99% coverage: max_seq_length >= {int(np.ceil(p99))}")
    print(f"  • For 100% coverage: max_seq_length >= {int(np.max(lengths_array))}")
    
    # Suggest practical values
    practical_95 = min([t for t in thresholds if t >= p95], default=int(np.ceil(p95)))
    practical_99 = min([t for t in thresholds if t >= p99], default=int(np.ceil(p99)))
    
    print("\n  Practical suggestions:")
    print(f"  • Balanced (95% coverage): --max-seq-length {practical_95}")
    print(f"  • Conservative (99% coverage): --max-seq-length {practical_99}")
    
    if np.max(lengths_array) > 1024:
        print("\n  ⚠️  Warning: Some sequences exceed 1024 tokens!")
        print("     Consider filtering or splitting long sequences.")
    
    print("=" * 60)
# ------------------------------------------------------------
# Show Top-5 Longest Problems
# ------------------------------------------------------------
    if longest_problems:
        print("\n" + "=" * 60)
        print(f"TOP {len(longest_problems)} LONGEST PROBLEMS")
        print("=" * 60)

        # Sort descending
        longest_sorted = sorted(longest_problems, key=lambda x: x[0], reverse=True)

        for rank, (plen, idx, ptext) in enumerate(longest_sorted, start=1):
            print(f"\n#{rank} | line={idx} | problem_tokens={plen}")

            preview = ptext.replace("\n", "\\n")
            if len(preview) > 500:
                preview = preview[:500] + " ... [truncated]"

            print("Problem preview:")
            print(preview)
        # ------------------------------------------------------------
    # Problem-only length distribution
    # ------------------------------------------------------------
    if problem_lengths:
        p_arr = np.array(problem_lengths)
        percentiles = [50, 75, 90, 95, 99, 99.5, 100]

        print("\n" + "=" * 60)
        print("PROBLEM-ONLY TOKEN LENGTH ANALYSIS")
        print("=" * 60)
        print(f"\nTotal problems analyzed: {len(problem_lengths)}")
        if problem_skipped > 0:
            print(f"Problems skipped (missing/invalid): {problem_skipped}")

        print("\nBasic Statistics (problem only):")
        print(f"  Min length:     {np.min(p_arr):>6} tokens")
        print(f"  Max length:     {np.max(p_arr):>6} tokens")
        print(f"  Mean length:    {np.mean(p_arr):>6.1f} tokens")
        print(f"  Median length:  {np.median(p_arr):>6.1f} tokens")
        print(f"  Std deviation:  {np.std(p_arr):>6.1f} tokens")

        print("\nPercentiles (problem only):")
        for p in percentiles:
            val = np.percentile(p_arr, p)
            print(f"  {p:>5.1f}th percentile: {val:>6.0f} tokens")

        print("\nCoverage by max_seq_length (problem only):")
        thresholds = [32, 64, 96, 128, 192, 256, 384, 512]
        for threshold in thresholds:
            count = np.sum(p_arr <= threshold)
            pct = count / len(p_arr) * 100
            truncated = len(p_arr) - count
            print(f"  max_seq_length={threshold:>4}: {count:>6}/{len(p_arr)} ({pct:>5.1f}%) | {truncated:>5} > threshold")

        # Optional: tiny text histogram
        print("\nHistogram (problem only, rough buckets):")
        buckets = [0, 16, 32, 64, 96, 128, 192, 256, 384, 512, 10**9]
        labels = [
            "  1-16", " 17-32", " 33-64", " 65-96", " 97-128",
            "129-192", "193-256", "257-384", "385-512", "513+"
        ]
        counts = []
        for lo, hi in zip(buckets[:-1], buckets[1:]):
            if lo == 0:
                c = np.sum((p_arr >= 1) & (p_arr <= hi))
            elif hi == 10**9:
                c = np.sum(p_arr >= lo + 1)
            else:
                c = np.sum((p_arr >= lo + 1) & (p_arr <= hi))
            counts.append(int(c))

        maxc = max(counts) if counts else 1
        for lab, c in zip(labels, counts):
            bar = "#" * int(40 * c / maxc) if maxc > 0 else ""
            print(f"  {lab}: {c:>6} {bar}")
    else:
        print("\n(No valid 'problem' fields found to analyze problem-only lengths.)")
        # ------------------------------------------------------------
    # Show longest examples
    # ------------------------------------------------------------
    if longest_heap:
        print("\n" + "=" * 60)
        print(f"LONGEST EXAMPLES (top {len(longest_heap)})")
        print("=" * 60)

        # Sort descending by length
        longest_sorted = sorted(longest_heap, key=lambda x: x[0], reverse=True)

        def preview(s, max_chars=400):
            s = s.replace("\n", "\\n")
            return s if len(s) <= max_chars else (s[:max_chars] + " ... [truncated]")

        for rank, (L, idx, text, toks) in enumerate(longest_sorted, start=1):
            print(f"\n#{rank} | line={idx} | length={L} tokens")
            print("Text preview:")
            print("  " + preview(text, 500))

            # Show token id slices
            head = toks[:30]
            tail = toks[-30:] if len(toks) > 30 else []
            print(f"Token IDs head(30): {head}")
            if tail:
                print(f"Token IDs tail(30): {tail}")

            # If decode is available, show decoded preview
            if hasattr(tokenizer, "decode"):
                try:
                    decoded = tokenizer.decode(toks)
                    print("Decoded preview:")
                    print("  " + preview(decoded, 500))
                except Exception:
                    pass
    print("\n" + "=" * 60)
    print("LONGEST PROBLEM ONLY")
    print("=" * 60)

    if max_problem_len > 0:
        print(f"Line index: {max_problem_idx}")
        print(f"Problem token length: {max_problem_len}")

        preview = max_problem_text.replace("\n", "\\n")
        if len(preview) > 500:
            preview = preview[:500] + " ... [truncated]"

        print("Problem preview:")
        print(preview)
    else:
        print("No problem field found.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze corpus sequence lengths to determine optimal max_seq_length"
    )
    
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to corpus file to analyze"
    )
    
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to trained tokenizer directory"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    
    parser.add_argument(
        "--corpus-type",
        type=str,
        choices=['foundational', 'instruction', 'auto'],
        default='auto',
        help="Type of corpus format (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect corpus type if needed
    corpus_type = args.corpus_type
    if corpus_type == 'auto':
        if 'instruction' in args.corpus_path.lower():
            corpus_type = 'instruction'
        else:
            corpus_type = 'foundational'
        print(f"Auto-detected corpus type: {corpus_type}")
    
    analyze_corpus_lengths(
        corpus_path=args.corpus_path,
        tokenizer_path=args.tokenizer_path,
        max_samples=args.max_samples,
        corpus_type=corpus_type
    )


if __name__ == "__main__":
    main()


