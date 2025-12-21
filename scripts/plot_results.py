"""
Plot evaluation results saved by bucket_evaluate for bm25_tiered.

Usage:
  python -m scripts.plot_results \
    --input <dev | eval1 | eval2> \
    --metric <MRR@10 | Recall@100 | NDCG@10 | NDCG@100> \
    [--type <overall | short | medium | long>] \
    [--output <output_png_filename>]

Notes:
- If --type is omitted, defaults to "overall".
- If --output is omitted, prints a table instead of saving a plot (sanity check).
- Ignores the file named "all".
- If --output is provided, saves to: plots/bm25_tiered/<split>/<output>
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from utils.config import BM25_TIERED_RESULTS_DIR, BM25_TIERED_PLOTS_DIR

METRICS = ["MRR@10", "Recall@100", "NDCG@10", "NDCG@100"]

TFILE_RE = re.compile(r"^t(\d+)$", re.IGNORECASE)
METRIC_LINE_RE = re.compile(r"^\s*([A-Za-z0-9@]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")
OVERALL_HEADER_RE = re.compile(r"=+\s*OVERALL PERFORMANCE\s*=+")
BY_LENGTH_HEADER_RE = re.compile(r"=+\s*PERFORMANCE BY QUERY LENGTH\s*=+")
BUCKET_HEADER_RE = re.compile(r"^\s*---\s*(SHORT|MEDIUM|LONG)\s+QUERIES\s*---\s*$", re.IGNORECASE)


def _list_run_files(split_dir: str) -> List[Tuple[int, str, str]]:
    """Return [(30, 't30', '/path/to/t30'), ...] sorted by threshold."""
    out: List[Tuple[int, str, str]] = []
    for name in os.listdir(split_dir):
        if name.startswith("."):
            continue
        if name.lower() == "all":
            continue
        match = TFILE_RE.match(name)
        if not match:
            continue
        path = os.path.join(split_dir, name)
        if os.path.isfile(path):
            out.append((int(match.group(1)), name.lower(), path))
    out.sort(key=lambda x: x[0])
    return out


def _parse_file(path: str) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      {
        "overall": {metric: val},
        "short": {metric: val},
        "medium": {metric: val},
        "long": {metric: val},
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    out: Dict[str, Dict[str, float]] = {"overall": {}, "short": {}, "medium": {}, "long": {}}

    # Overall block
    in_overall = False
    for line in lines:
        if OVERALL_HEADER_RE.search(line):
            in_overall = True
            continue
        if in_overall:
            if line.strip().startswith("====") and "OVERALL PERFORMANCE" not in line:
                break
            metric_line = METRIC_LINE_RE.match(line)
            if metric_line:
                out["overall"][metric_line.group(1)] = float(metric_line.group(2))

    # By-length block
    in_by_length = False
    bucket: Optional[str] = None
    for line in lines:
        if BY_LENGTH_HEADER_RE.search(line):
            in_by_length = True
            bucket = None
            continue
        if not in_by_length:
            continue

        bucket_header = BUCKET_HEADER_RE.match(line)
        if bucket_header:
            bucket = bucket_header.group(1).lower()
            continue

        if bucket:
            metric_line = METRIC_LINE_RE.match(line)
            if metric_line:
                out[bucket][metric_line.group(1)] = float(metric_line.group(2))

    return out


def _title(split: str, kind: str, metric: str) -> str:
    if kind == "overall":
        return f"{split} — overall — {metric}"
    return f"{split} — {kind} queries — {metric}"


def _print_table(split: str, kind: str, metric: str, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    print(f"\n{_title(split, kind, metric)}\n")
    print(f"{'run':>3}  {metric:>10}")
    for run, metrics in rows:
        v = metrics.get(metric, None)
        val_str = f"{v:>10.4f}" if v is not None else f"{'NA':>10}"
        print(f"{run:>3}  {val_str}")


def _plot(split: str, kind: str, metric: str, rows: List[Tuple[str, Dict[str, float]]], out_path: str) -> None:
    x = [run for run, _ in rows]
    y = [metrics.get(metric, None) for _, metrics in rows]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(_title(split, kind, metric))
    plt.xlabel("threshold")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot in {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Split name, e.g. dev, eval1, eval2")
    ap.add_argument("--metric", choices=METRICS, required=True)
    ap.add_argument("--type", choices=["overall", "short", "medium", "long"], default="overall")
    ap.add_argument("--output", required=False, help="Output PNG filename. If omitted, prints a table.")
    args = ap.parse_args()

    split = args.input
    kind = args.type
    metric = args.metric

    split_dir = os.path.join(BM25_TIERED_RESULTS_DIR, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Input folder not found: {split_dir}")

    run_files = _list_run_files(split_dir)
    if not run_files:
        raise RuntimeError(f"No tXX files found in {split_dir}.")

    rows: List[Tuple[str, Dict[str, float]]] = []
    for _, run, path in run_files:
        parsed = _parse_file(path)
        rows.append((run, parsed.get(kind, {})))

    if not args.output:
        _print_table(split, kind, metric, rows)
        return

    out_path = os.path.join(BM25_TIERED_PLOTS_DIR, split, args.output)
    _plot(split, kind, metric, rows, out_path)


if __name__ == "__main__":
    main()