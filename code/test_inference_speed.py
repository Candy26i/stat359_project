import argparse
import time
import torch

from instructor.final_project.arithmetic_llm.evaluator import ModelEvaluator


def benchmark(
    adapter_path: str,
    base_checkpoint_path: str,
    tokenizer_path: str,
    device: str,
    num_samples: int,
    batch_size: int,
    max_gen_length: int,
    warmup_batches: int,
):
    # Create evaluator with NON-MERGED adapter
    # model_path = adapter checkpoint (lora_adapter.pt)
    # base_checkpoint_path = foundational best_model.pt
    evaluator = ModelEvaluator(
        model_path=adapter_path,
        tokenizer_path=tokenizer_path,
        base_checkpoint_path=base_checkpoint_path,
        device=device,
    )

    # Build fixed prompts so each run is comparable
    prompts = [f"Evaluate: 5 + 7 - (10 - 3) <think>" for _ in range(num_samples)]

    # Warmup (important for CUDA kernel caching)
    if warmup_batches > 0:
        for _ in range(warmup_batches):
            _ = evaluator._generate_batch(prompts[:batch_size], max_length=max_gen_length)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed generation
    start = time.time()

    total_gen_tokens = 0
    total_samples_done = 0

    for i in range(0, num_samples, batch_size):
        batch = prompts[i : i + batch_size]
        outs = evaluator._generate_batch(batch, max_length=max_gen_length)

        # Count generated tokens (approx): total_len - prompt_len
        # This matches the logic in your evaluator.evaluate()
        for prompt, out_text in zip(batch, outs):
            prompt_ids = evaluator.tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = evaluator.tokenizer.encode(out_text, add_special_tokens=False)
            gen_len = max(0, len(full_ids) - len(prompt_ids))
            total_gen_tokens += gen_len
            total_samples_done += 1

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    total_time = end - start

    samples_per_sec = total_samples_done / total_time if total_time > 0 else 0.0
    tokens_per_sec = total_gen_tokens / total_time if total_time > 0 else 0.0

    print("\n===== NON-MERGED LoRA Benchmark =====")
    print(f"Adapter: {adapter_path}")
    print(f"Base:    {base_checkpoint_path}")
    print(f"Device:  {device}")
    print(f"Samples: {total_samples_done}")
    print(f"Batch:   {batch_size}")
    print(f"MaxLen:  {max_gen_length}")
    print("------------------------------------")
    print(f"Total time (s):      {total_time:.4f}")
    print(f"Samples/sec:         {samples_per_sec:.4f}")
    print(f"Total gen tokens:    {total_gen_tokens}")
    print(f"Generated tokens/sec:{tokens_per_sec:.4f}")
    print("====================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True, help="Path to lora_adapter.pt (NOT merged_model.pt)")
    ap.add_argument("--base-checkpoint", required=True, help="Path to foundational best_model.pt")
    ap.add_argument("--tokenizer-path", default="data/tokenizer", help="Tokenizer directory (default: data/tokenizer)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-gen-length", type=int, default=256)
    ap.add_argument("--warmup-batches", type=int, default=2)
    args = ap.parse_args()

    benchmark(
        adapter_path=args.adapter_path,
        base_checkpoint_path=args.base_checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_gen_length=args.max_gen_length,
        warmup_batches=args.warmup_batches,
    )


if __name__ == "__main__":
    main()