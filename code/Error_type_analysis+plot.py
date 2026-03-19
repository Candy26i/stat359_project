#!/usr/bin/env python3
"""
No-argparse version: Error type vs prompt token length.

Edits:
- INPUT_PATH / OUT_DIR / PREFIX / BIN_STEP / colors below, then run:
    poetry run python instructor/final_project/arithmetic_llm/测试plot.py

This script bins samples by prompt_len and plots a stacked bar chart of:
  1) Unparsable (parseable == False)
  2) Wrong reasoning (parseable == True and correct == False)
  3) Correct (correct == True)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG (EDIT THESE)
# =========================
# INPUT_PATH = r"evaluation_results\sample_outputs_20260227_205212.json"
# INPUT_PATH = r"C:\Users\madis\Desktop\stat_359\stat359\evaluation_results\sample_outputs_20260303_004358.json"
# INPUT_PATH = r"C:\Users\madis\Desktop\stat_359\stat359\evaluation_results\sample_outputs_20260302_221958.json" 
# INPUT_PATH = r"C:\Users\madis\Desktop\stat_359\stat359\evaluation_results\sample_outputs_20260302_223656.json" $32
INPUT_PATH = r"C:\Users\madis\Desktop\stat_359\stat359\evaluation_results\sample_outputs_20260302_220556.json" #8
OUT_DIR = r"evaluation_results"
PREFIX = "error_type_vs_prompt_len_r8"  # output filename prefix

AUTO_BINS = True                   # True -> auto bins up to max(prompt_len)
BIN_STEP = 10                      # bin width if AUTO_BINS=True
MANUAL_BINS = None                 # e.g. [0,10,20,30,40,50,60,80,100,150,200,300]

# Optional: also write a per-bin summary table to stdout
PRINT_TABLE = True

# Colors (matplotlib named colors)
COLOR_UNPARSABLE = "tab:red"
COLOR_WRONG = "tab:orange"
COLOR_CORRECT = "tab:green"
# =========================


def load_records(path: str):
    """Load JSON list / JSON dict with nested list / JSONL into a list[dict]."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    # JSON list
    if text[0] == "[":
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []

    # JSON dict with nested list
    if text[0] == "{":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ["samples", "examples", "records", "data", "outputs", "results"]:
                v = obj.get(k)
                if isinstance(v, list):
                    return v
        return []

    # JSONL fallback
    records = []
    with open(path, "r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def extract_categories(records):
    """
    Returns:
      x: np.ndarray[int] prompt_len
      cats: np.ndarray[int] with values:
        0 = unparsable
        1 = wrong reasoning (parseable but incorrect)
        2 = correct
    Requires keys:
      - prompt_len
      - parseable (optional; defaults False)
      - correct (optional; defaults False)
    """
    xs = []
    cats = []

    for r in records:
        if "prompt_len" not in r:
            continue
        try:
            x = int(r["prompt_len"])
        except Exception:
            continue

        parseable = bool(r.get("parseable", False))
        correct = bool(r.get("correct", False))

        if not parseable:
            cat = 0
        elif not correct:
            cat = 1
        else:
            cat = 2

        xs.append(x)
        cats.append(cat)

    return np.array(xs, dtype=int), np.array(cats, dtype=int)


def make_auto_bins(x: np.ndarray, step: int = 10, max_bins: int = 30):
    """Create bins edges [0, step, 2*step, ...] up to cover max(x), with a cap on number of bins."""
    xmax = int(x.max())
    est_bins = int(np.ceil(xmax / step))
    if est_bins > max_bins:
        step = int(np.ceil(xmax / max_bins))
        nice = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
        step = next((n for n in nice if n >= step), step)

    bins = list(range(0, xmax + step, step))
    if bins[-1] < xmax:
        bins.append(xmax)
    if bins[-1] < xmax:
        bins[-1] = xmax
    return bins


def bin_stats_multiclass(x, cats, bins):
    """
    For each bin (lo, hi], compute % in each category and count.
    Returns:
      labels: list[str]
      u_rate, w_rate, c_rate: list[float] (percentages)
      cnt: list[int]
    """
    bins = np.array(bins, dtype=int)
    labels = []
    u_rate, w_rate, c_rate = [], [], []
    cnt = []

    for i in range(1, len(bins)):
        lo, hi = int(bins[i - 1]), int(bins[i])
        m = (x > lo) & (x <= hi)
        n = int(m.sum())

        labels.append(f"{lo+1}-{hi}")
        cnt.append(n)

        if n == 0:
            u_rate.append(np.nan)
            w_rate.append(np.nan)
            c_rate.append(np.nan)
        else:
            sub = cats[m]
            u_rate.append(100.0 * np.mean(sub == 0))
            w_rate.append(100.0 * np.mean(sub == 1))
            c_rate.append(100.0 * np.mean(sub == 2))

    return labels, u_rate, w_rate, c_rate, cnt


def safe_arr(a_list):
    """Convert list to float ndarray, replacing nan with 0 for stacking."""
    arr = np.array(a_list, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def print_summary(labels, u_rate, w_rate, c_rate, cnt):
    """Pretty-ish table in stdout."""
    print("\nPer-bin breakdown (%):")
    print(f"{'bin':>12} | {'count':>6} | {'unparsable':>10} | {'wrong':>7} | {'correct':>8}")
    print("-" * 60)
    for lab, n, u, w, c in zip(labels, cnt, u_rate, w_rate, c_rate):
        # Keep NaNs as 'nan' if present
        def fmt(v):
            return "  nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:6.2f}"
        print(f"{lab:>12} | {n:6d} | {fmt(u):>10} | {fmt(w):>7} | {fmt(c):>8}")
    print("-" * 60)


def main():
    print(f"[plot] reading: {INPUT_PATH}")
    records = load_records(INPUT_PATH)
    print(f"[plot] loaded records: {len(records)}")

    x, cats = extract_categories(records)
    print(f"[plot] usable samples (have prompt_len): {len(x)}")

    if len(x) == 0:
        if records:
            print("[plot] first record keys:", list(records[0].keys()))
        raise SystemExit("No usable samples. Need key: prompt_len (and ideally parseable/correct).")

    xmax = int(x.max())
    overall_unparsable = 100.0 * np.mean(cats == 0)
    overall_wrong = 100.0 * np.mean(cats == 1)
    overall_correct = 100.0 * np.mean(cats == 2)

    print(f"[plot] overall: unparsable={overall_unparsable:.2f}%, wrong={overall_wrong:.2f}%, correct={overall_correct:.2f}%")
    print(f"[plot] prompt_len summary: min={x.min()}, median={np.median(x):.1f}, mean={x.mean():.1f}, max={xmax}")

    # bins
    if MANUAL_BINS is not None and len(MANUAL_BINS) >= 2:
        bins = list(MANUAL_BINS)
        if bins[-1] < xmax:
            bins.append(xmax)
            print(f"[plot] extended manual bins to cover max: last_edge={bins[-1]}")
    else:
        if AUTO_BINS:
            bins = make_auto_bins(x, step=BIN_STEP)
            print(f"[plot] auto bins cover up to: {bins[-1]} (max={xmax})")
        else:
            bins = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
            if bins[-1] < xmax:
                bins.append(xmax)

    labels, u_rate, w_rate, c_rate, cnt = bin_stats_multiclass(x, cats, bins)

    if PRINT_TABLE:
        print_summary(labels, u_rate, w_rate, c_rate, cnt)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{PREFIX}.png")

    idx = np.arange(len(labels))

    # Convert to arrays for stacking (NaN -> 0)
    u = safe_arr(u_rate)
    w = safe_arr(w_rate)
    c = safe_arr(c_rate)

    fig, ax = plt.subplots()

    ax.bar(idx, u, label="Unparsable", color=COLOR_UNPARSABLE)
    ax.bar(idx, w, bottom=u, label="Wrong reasoning", color=COLOR_WRONG)
    ax.bar(idx, c, bottom=(u + w), label="Correct", color=COLOR_CORRECT)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Error Type vs Prompt Token Length")

    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"\n[plot] saved: {out_path}")


if __name__ == "__main__":
    main()