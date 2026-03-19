#!/usr/bin/env python3
"""
No-argparse version.
Edit INPUT_PATH / OUT_DIR / PREFIX / BIN_STEP / LINE_COLOR below, then run:
    poetry run python instructor/final_project/arithmetic_llm/测试plot.py
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
INPUT_PATH = r"C:\Users\madis\Desktop\stat_359\stat359\evaluation_results\sample_outputs_20260302_221958.json"
OUT_DIR = r"evaluation_results"
PREFIX = "accuracy_vs_prompt_len_r2"

REQUIRE_PARSEABLE = False          # True -> only parseable==True
AUTO_BINS = True                   # True -> auto bins up to max(prompt_len)
BIN_STEP = 10                      # bin width if AUTO_BINS=True
MANUAL_BINS = None                 # e.g. [0,10,20,30,40,50,60,80,100,150,200,300]

LINE_COLOR = "tab:orange"          # <-- change line color here
# =========================


def load_records(path: str):
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


def extract_xy(records, require_parseable=False):
    xs, ys = [], []
    for r in records:
        if require_parseable and not bool(r.get("parseable", False)):
            continue
        if "prompt_len" not in r or "correct" not in r:
            continue
        try:
            x = int(r["prompt_len"])
        except Exception:
            continue

        c = r["correct"]
        if isinstance(c, (int, float)) and c in (0, 1):
            y = bool(c)
        else:
            y = bool(c)

        xs.append(x)
        ys.append(y)

    return np.array(xs, dtype=int), np.array(ys, dtype=bool)


def make_auto_bins(x: np.ndarray, step: int = 10, max_bins: int = 30):
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


def bin_stats(x, y, bins):
    bins = np.array(bins, dtype=int)
    labels, acc, cnt = [], [], []

    for i in range(1, len(bins)):
        lo, hi = int(bins[i - 1]), int(bins[i])
        m = (x > lo) & (x <= hi)
        n = int(m.sum())
        labels.append(f"{lo+1}-{hi}")
        cnt.append(n)
        acc.append(np.nan if n == 0 else 100.0 * y[m].mean())

    last = int(bins[-1])
    m = x > last
    n = int(m.sum())
    if n > 0:
        labels.append(f">{last}")
        cnt.append(n)
        acc.append(100.0 * y[m].mean())

    return labels, acc, cnt


def main():
    print(f"[plot] reading: {INPUT_PATH}")
    records = load_records(INPUT_PATH)
    print(f"[plot] loaded records: {len(records)}")

    x, y = extract_xy(records, require_parseable=REQUIRE_PARSEABLE)
    print(f"[plot] usable samples (have prompt_len & correct): {len(x)}")

    if len(x) == 0:
        if records:
            print("[plot] first record keys:", list(records[0].keys()))
        raise SystemExit("No usable samples. Need keys: prompt_len and correct.")

    xmax = int(x.max())
    print(f"[plot] overall accuracy: {100.0 * y.mean():.2f}%")
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
            # fallback default bins
            bins = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
            if bins[-1] < xmax:
                bins.append(xmax)

    labels, acc, cnt = bin_stats(x, y, bins)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{PREFIX}.png")

    idx = np.arange(len(labels))
    fig, ax1 = plt.subplots()

    ax1.bar(idx, acc, label="Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_title("Accuracy vs Prompt Token Length" + (" (parseable only)" if REQUIRE_PARSEABLE else ""))

    ax2 = ax1.twinx()
    ax2.plot(idx, cnt, marker="o", color=LINE_COLOR, label="Count")
    ax2.set_ylabel("Count")

    # legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()