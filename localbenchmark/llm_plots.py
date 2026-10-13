"""
Plots for llm_coding_benchmark.py results.

Standalone:
    python plot_results.py --results coding_benchmark_results.csv --out plots

Also called automatically at the end of the benchmark (disable with --no-plots).
If the CSV contains several --fill ratios, one sub-folder is written per ratio,
because runs with different fill are not comparable with each other.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NUMERIC = [
    "context_actual", "context_requested", "fill_ratio", "ttft",
    "generation_time", "decode_time", "prompt_tokens", "completion_tokens",
    "decode_tokens_per_sec", "gpu_memory_mb",
]

# Default planned windows (kept in sync with llm_coding_benchmark.CONTEXTS).
# Snap targets are always extended with sizes observed in the results file,
# so any context the user adds (e.g. 80_000) appears in plots 09/10.
PLANNED_CONTEXTS = [50_000, 80_000, 100_000, 120_000, 150_000, 180_000]

# Contexts below this threshold get their own (worst) rank tier.
LOW_CONTEXT_THRESHOLD = 35_000
LOW_CONTEXT_BIN = 0  # sentinel used in context_bin


def discover_context_bins(contexts, extras=None):
    """
    Build snap targets from observed context sizes + known planned windows.

    Each observed size is rounded to the nearest 1k so tiny server differences
    (49992 vs 50000) collapse, while distinct windows like 50k and 80k stay apart.
    """
    bins = set(PLANNED_CONTEXTS)
    if extras:
        bins.update(int(x) for x in extras)
    for c in contexts:
        c = float(c)
        if c < LOW_CONTEXT_THRESHOLD:
            continue
        bins.add(int(round(c / 1000.0) * 1000))
    return sorted(bins)


def snap_context(ctx, planned=None):
    """Map an actual context size to the nearest known window (or LOW_CONTEXT_BIN)."""
    ctx = float(ctx)
    if ctx < LOW_CONTEXT_THRESHOLD:
        return LOW_CONTEXT_BIN
    if planned is None:
        planned = PLANNED_CONTEXTS
    planned = np.asarray(list(planned), dtype=float)
    if planned.size == 0:
        return int(round(ctx / 1000.0) * 1000)
    return int(planned[np.argmin(np.abs(planned - ctx))])


# =============================================================================
# DATA
# =============================================================================

def wilson(passes, n, z=1.96):
    """95% Wilson interval, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = passes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


RAW_REQUIRED = {"experiment_id", "problem_id", "passed", "model", "cache_label",
                "context_actual", "fill_ratio"}


def is_raw_results(path):
    """True for a per-problem results CSV; False for summaries and other tables."""
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return RAW_REQUIRED.issubset(cols)


def find_raw_results(folder="."):
    """Every per-problem results CSV in a folder and in its benchmark sub-folders
    (humaneval/, mbpp/, swebench/, suite/, dryrun/), whatever the file name."""
    root = Path(folder)
    candidates = list(root.glob("*.csv")) + list(root.glob("*/*.csv"))
    return sorted(p for p in candidates if is_raw_results(p))


def resolve_inputs(results=None, folder="."):
    """
    Return a list of (per_problem_csv, subfolder_name_or_None).

    - No --results: every per-problem CSV in the folder, one subfolder each.
    - --results <raw file>: that file, written straight into --out.
    - --results <name>_summary.csv: its per-problem sibling <name>.csv.
    """
    if results is None:
        return [(p, p.stem) for p in find_raw_results(folder)]

    p = Path(results)
    if not p.exists():
        raise FileNotFoundError(f"results file not found: {p}")
    if is_raw_results(p):
        return [(p, None)]

    if p.stem.endswith("_summary"):
        sibling = p.with_name(p.stem[: -len("_summary")] + ".csv")
        if sibling.exists() and is_raw_results(sibling):
            print(f"{p.name} is an aggregated summary; plotting from {sibling.name}.")
            return [(sibling, None)]
        raise FileNotFoundError(
            f"{p.name} is an aggregated summary. The plots need the per-problem file "
            f"{sibling.name}, which is not in {p.parent.resolve()}. "
            "localbenchmark writes both files; copy the per-problem CSV next to the summary.")

    raise ValueError(
        f"{p.name} is not a per-problem results file "
        f"(it needs the columns: {', '.join(sorted(RAW_REQUIRED))}).")


def load_results(path):
    df = pd.read_csv(path)
    missing = RAW_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"{Path(path).name} is not a per-problem results file "
            f"(missing columns: {', '.join(sorted(missing))})")
    df = df.dropna(subset=["experiment_id", "problem_id"])

    # If a problem was re-run (e.g. a retried LLM error), keep the latest row.
    df = df.drop_duplicates(["experiment_id", "problem_id"], keep="last").copy()

    df["passed"] = df["passed"].astype(str).str.lower().eq("true")
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    short = df["model"].astype(str).str.split("/").str[-1]
    if df["cache_label"].nunique() > 1:
        df["series"] = short + " [" + df["cache_label"].astype(str) + "]"
    else:
        df["series"] = short
    return df


def weighted_tps(g):
    """Total tokens / total decode time (not the mean of per-problem ratios)."""
    v = g[(g["decode_time"] > 0) & g["completion_tokens"].notna()]
    if v.empty:
        return float("nan")
    return v["completion_tokens"].sum() / v["decode_time"].sum()


def agg_by_context(df):
    # Prefer the requested window when present so plots track the benchmark
    # CONTEXTS list (including newly added sizes like 80k). Fall back to actual.
    ctx_col = "context_requested" if "context_requested" in df.columns else "context_actual"
    rows = []
    for (series, ctx), g in df.groupby(["series", ctx_col]):
        n, p = len(g), int(g["passed"].sum())
        lo, hi = wilson(p, n)
        rows.append({
            "series": series,
            "context": int(ctx),
            "n": n,
            "passed": p,
            "pass_rate": 100 * p / n,
            "ci_low": lo,
            "ci_high": hi,
            "tps": weighted_tps(g),
            "ttft": g["ttft"].median(),
            "prompt_tokens": g["prompt_tokens"].mean(),
            "gpu_mb": g["gpu_memory_mb"].max(),
        })
    return pd.DataFrame(rows).sort_values(["series", "context"])


def leaderboard(df):
    rows = []
    for series, g in df.groupby("series"):
        n, p = len(g), int(g["passed"].sum())
        lo, hi = wilson(p, n)
        rows.append({
            "series": series,
            "n": n,
            "pass_rate": 100 * p / n,
            "ci_low": lo,
            "ci_high": hi,
            "tps": weighted_tps(g),
            "ttft_median": g["ttft"].median(),
            "gpu_peak_mb": g["gpu_memory_mb"].max(),
            "truncated": int((g["error_type"] == "TRUNCATED").sum()),
            "llm_errors": int((g["error_type"] == "LLM_ERROR").sum()),
        })
    return pd.DataFrame(rows).sort_values("pass_rate", ascending=False)


def trend_table(agg):
    """Linear trend of tps / pass rate / ttft against context size."""
    rows = []
    for series, g in agg.groupby("series"):
        g = g.sort_values("context")
        if len(g) < 2:
            continue
        x = g["context"].to_numpy(dtype=float) / 100_000
        row = {"series": series, "contexts_tested": len(g)}
        for col in ("tps", "pass_rate", "ttft"):
            y = g[col].to_numpy(dtype=float)
            m = ~np.isnan(y)
            if m.sum() >= 2:
                row[f"{col}_slope_per_100k_ctx"] = np.polyfit(x[m], y[m], 1)[0]
                first, last = y[m][0], y[m][-1]
                row[f"{col}_change_pct_first_to_last"] = (
                    100 * (last - first) / first if first else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def style_map(series_list):
    cmap = plt.get_cmap("tab10")
    markers = "osD^vP*X"
    return {
        s: (cmap(i % 10), markers[i % len(markers)])
        for i, s in enumerate(series_list)
    }


def set_ctx_axis(ax, contexts):
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c / 1000:g}k" for c in contexts],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Context window (tokens)")


def legend_outside(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
              frameon=False)


def save(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_ranking(lb, out_dir):
    """Overall score per model (all contexts pooled), with CI and speed."""
    lb = lb.sort_values("pass_rate")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(lb) + 2))
    y = np.arange(len(lb))
    err = np.array([lb["pass_rate"] - lb["ci_low"], lb["ci_high"] - lb["pass_rate"]])
    ax.barh(y, lb["pass_rate"], xerr=err, color="#4C78A8", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(lb["series"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pass rate (%) with 95% CI, all contexts pooled")
    ax.set_title("Overall ranking by pass rate")
    for yi, (_, r) in zip(y, lb.iterrows()):
        tps = "n/a" if np.isnan(r["tps"]) else f"{r['tps']:.0f} tok/s"
        ax.text(min(r["ci_high"] + 1, 99), yi, f"{r['pass_rate']:.1f}%  |  {tps}",
                va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    save(fig, out_dir, "01_ranking_pass_rate.png")


def plot_pass_vs_context(agg, styles, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, g in agg.groupby("series"):
        c, m = styles[s]
        ax.plot(g["context"], g["pass_rate"], marker=m, color=c, label=s)
        ax.fill_between(g["context"], g["ci_low"], g["ci_high"], color=c, alpha=0.12)
    set_ctx_axis(ax, sorted(agg["context"].unique()))
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pass rate vs context window (shaded: 95% CI)")
    ax.grid(alpha=0.3)
    legend_outside(ax)
    save(fig, out_dir, "02_pass_rate_vs_context.png")


def plot_heatmap(agg, out_dir):
    pivot = agg.pivot(index="series", columns="context", values="pass_rate")
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 3,
                                    0.6 * len(pivot) + 2))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c / 1000:g}k" for c in pivot.columns],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Context window (tokens)")
    ax.set_title("Pass rate (%) by model and context")
    fig.colorbar(im, ax=ax, label="Pass rate (%)")
    save(fig, out_dir, "03_pass_rate_heatmap.png")


def plot_tps_vs_context(agg, styles, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, g in agg.groupby("series"):
        c, m = styles[s]
        ax.plot(g["context"], g["tps"], marker=m, color=c, label=s)
    set_ctx_axis(ax, sorted(agg["context"].unique()))
    ax.set_ylabel("Decode speed (tokens/s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Decode speed vs context window")
    ax.grid(alpha=0.3)
    legend_outside(ax)
    save(fig, out_dir, "04_tps_vs_context.png")


def plot_tps_retention(agg, trend, styles, out_dir):
    """Speed relative to each model's smallest context: shows the slowdown."""
    change = {}
    if not trend.empty and "tps_change_pct_first_to_last" in trend:
        change = dict(zip(trend["series"], trend["tps_change_pct_first_to_last"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, g in agg.groupby("series"):
        g = g.sort_values("context")
        valid = g["tps"].dropna()
        if valid.empty:
            continue
        base = valid.iloc[0]
        c, m = styles[s]
        label = s
        if s in change and not np.isnan(change[s]):
            label += f" ({change[s]:+.0f}%)"
        ax.plot(g["context"], 100 * g["tps"] / base, marker=m, color=c, label=label)
    ax.axhline(100, color="black", lw=0.8, ls="--")
    set_ctx_axis(ax, sorted(agg["context"].unique()))
    ax.set_ylabel("Speed relative to smallest context (%)")
    ax.set_title("Slowdown trend: tokens/s relative to smallest context\n"
                 "(legend: change from smallest to largest context)")
    ax.grid(alpha=0.3)
    legend_outside(ax)
    save(fig, out_dir, "05_tps_retention.png")


def plot_ttft_vs_context(agg, styles, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, g in agg.groupby("series"):
        c, m = styles[s]
        ax.plot(g["context"], g["ttft"], marker=m, color=c, label=s)
    set_ctx_axis(ax, sorted(agg["context"].unique()))
    ax.set_ylabel("Median time to first token (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("Prompt-processing latency vs context window")
    ax.grid(alpha=0.3, which="both")
    legend_outside(ax)
    save(fig, out_dir, "06_ttft_vs_context.png")


def pareto_front(points):
    """Non-dominated points when maximising both coordinates."""
    front = []
    for i, (x, y) in enumerate(points):
        dominated = any(
            (x2 >= x and y2 >= y) and (x2 > x or y2 > y)
            for j, (x2, y2) in enumerate(points) if j != i
        )
        if not dominated:
            front.append((x, y))
    return sorted(front)


def plot_tradeoff(agg, styles, out_dir):
    """Score vs speed. Top-right is best; black line is the Pareto front."""
    d = agg.dropna(subset=["tps"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    cmin, cmax = d["context"].min(), d["context"].max()
    # Deterministic per-series jitter so overlapping points separate slightly.
    # Pareto front stays on true (unjittered) coordinates.
    rng = np.random.default_rng(0)
    series_list = list(d.groupby("series").groups)
    tps_span = max(d["tps"].max() - d["tps"].min(), 1.0)
    rate_span = max(d["pass_rate"].max() - d["pass_rate"].min(), 1.0)
    jx_scale = 0.015 * tps_span
    jy_scale = 0.015 * rate_span

    texts = []
    for i, s in enumerate(series_list):
        g = d[d["series"] == s].sort_values("context")
        c, m = styles[s]
        jx = rng.normal(0, jx_scale, len(g))
        jy = rng.normal(0, jy_scale, len(g))
        x = g["tps"].to_numpy(dtype=float) + jx
        y = g["pass_rate"].to_numpy(dtype=float) + jy
        ax.plot(x, y, color=c, alpha=0.4, lw=1)
        sizes = 40 + 160 * (g["context"] - cmin) / max(1, cmax - cmin)
        ax.scatter(x, y, s=sizes, color=c, marker=m,
                   edgecolor="black", linewidth=0.5, label=s, zorder=3)
        for k, (xi, yi, (_, row)) in enumerate(zip(x, y, g.iterrows())):
            label = f"{row['context'] / 1000:g}k"
            texts.append(ax.text(xi, yi, label, fontsize=7))

    # Prefer adjustText when installed; otherwise stagger offsets.
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
            expand_points=(1.2, 1.4),
            force_text=(0.3, 0.5),
        )
    except ImportError:
        for k, t in enumerate(texts):
            # Fan labels around the point so stacked ones spread out.
            angle = (k % 8) * (2 * np.pi / 8)
            t.set_position((
                t.get_position()[0] + 0.012 * tps_span * np.cos(angle),
                t.get_position()[1] + 0.012 * rate_span * np.sin(angle),
            ))

    front = pareto_front(list(zip(d["tps"], d["pass_rate"])))
    if len(front) > 1:
        ax.step([p[0] for p in front], [p[1] for p in front], where="post",
                color="black", ls="--", lw=1, label="Pareto front")

    ax.set_xlabel("Decode speed (tokens/s)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Quality vs speed (marker size = context window)")
    ax.grid(alpha=0.3)
    legend_outside(ax)
    save(fig, out_dir, "07_score_vs_speed.png")


def plot_gpu_vs_context(agg, styles, out_dir):
    d = agg.dropna(subset=["gpu_mb"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, g in d.groupby("series"):
        c, m = styles[s]
        ax.plot(g["context"], g["gpu_mb"] / 1024, marker=m, color=c, label=s)
    set_ctx_axis(ax, sorted(d["context"].unique()))
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("GPU memory vs context window")
    ax.grid(alpha=0.3)
    legend_outside(ax)
    save(fig, out_dir, "08_gpu_memory_vs_context.png")


_RANK_COLS = ["rank_pass", "rank_tps", "rank_ttft", "rank_gpu", "rank_ctx"]

# Pass rate and decode speed weighted higher; the rest equal weight.
RANK_WEIGHTS = {
    "rank_tps":  1.5,
    "rank_ctx":  1.5,
    "rank_pass": 1.0,
    "rank_ttft": 1.0,
    "rank_gpu":  1.0,
}


# Real metric columns shown in the heatmap cells (colour still uses ranks).
_VALUE_COLS_CTX = ["pass_rate_adj", "tps", "ttft", "gpu_mb", "context_bin"]
_VALUE_COLS_MODEL = ["pass_rate_adj", "tps", "ttft", "gpu_mb", "max_context"]


def _aggregate_ranks(ranks_df, how="sum"):
    """
    Combine per-metric ranks into a single score (lower = better).

    Pass rate and decode speed use weight 2; TTFT, GPU, context use weight 1.
    - sum:    Σ (rank × weight)
    - mean:   Σ (rank × weight) / Σ weight   (weighted mean)
    - median: unweighted median (weights do not apply)
    """
    how = (how or "sum").lower()
    w = np.array([RANK_WEIGHTS[c] for c in ranks_df.columns], dtype=float)
    weighted = ranks_df.to_numpy(dtype=float) * w

    if how == "mean":
        return pd.Series(weighted.sum(axis=1) / w.sum(), index=ranks_df.index)
    if how == "median":
        return ranks_df.median(axis=1)
    # default: weighted sum
    return pd.Series(weighted.sum(axis=1), index=ranks_df.index)


def _pct_score_higher_better(values, valid):
    """
    Score from % of the best value when higher is better (lower score = better).

    Used for decode speed and context size.

    best → 1.0
    half of best → 51.0
    example: best=229, other=60 → 1 + 100*(1 - 60/229) ≈ 74.8
    example: best=180k, other=50k → 1 + 100*(1 - 50/180) ≈ 73.2

    Incomplete rows (valid=False) get one step worse than the worst valid score.
    """
    scores = pd.Series(101.0, index=values.index, dtype=float)
    if not valid.any():
        return scores
    v = values.loc[valid].astype(float)
    best = v.max()
    if not np.isfinite(best) or best <= 0:
        scores.loc[valid] = 1.0
    else:
        scores.loc[valid] = 1.0 + 100.0 * (1.0 - v / best)
    worst_valid = float(scores.loc[valid].max())
    scores.loc[~valid] = worst_valid + 1.0
    return scores


def _assign_ranks(d, valid, metrics, rank_agg="sum"):
    """
    Rank only rows where `valid` is True. Incomplete rows get worst rank
    (n_valid + 1) on every metric so they sort to the bottom without
    affecting the ranking of complete rows.

    metrics: list of (source_col, rank_col, ascending)
      ascending=False → higher source value is better (rank 1)

    rank_agg: how to combine per-metric ranks into rank_sum —
              "sum" | "mean" | "median"
    """
    n_valid = int(valid.sum())
    worst = n_valid + 1
    for src, rank_col, ascending in metrics:
        d[rank_col] = float(worst)
        if n_valid > 0:
            d.loc[valid, rank_col] = (
                d.loc[valid, src].rank(ascending=ascending, method="average")
            )
    d["rank_sum"] = _aggregate_ranks(d[_RANK_COLS], rank_agg)
    d["_incomplete"] = ~valid
    return d.sort_values(["_incomplete", "rank_sum"]).drop(columns=["_incomplete"])


def rank_sum_table(agg, rank_agg="sum"):
    """
    Rank each (series, context) on five metrics and combine the ranks.

    Rank 1 = best. Lower combined score is better overall.

    Metrics (direction):
      pass_rate_adj  higher better  — aborted runs: unrun problems count as fail
      tps            higher better
      ttft           lower better
      gpu_mb         lower better
      context        higher better  — larger context *size* is better

    rank_agg: "sum" | "mean" | "median" — how ranks are combined.

    Rows with NaN decode speed or TTFT are excluded from ranking and
    placed at the bottom.
    """
    d = agg.copy()
    # Snap targets = planned defaults ∪ every size seen in this results file
    # (so a newly added context like 80k is not collapsed into 50k/100k).
    planned = discover_context_bins(d["context"])
    d["context_bin"] = d["context"].map(lambda c: snap_context(c, planned))
    # Merge rows that land on the same bin (e.g. 119872 and 120000 → 120k).
    d = (
        d.groupby(["series", "context_bin"], as_index=False)
        .agg(
            n=("n", "sum"),
            passed=("passed", "sum"),
            tps=("tps", "mean"),
            ttft=("ttft", "mean"),
            gpu_mb=("gpu_mb", "max"),
            context=("context", "max"),
        )
    )
    d["expected_n"] = d.groupby("context_bin")["n"].transform("max")
    d["pass_rate_adj"] = 100.0 * d["passed"] / d["expected_n"].clip(lower=1)

    valid = d["tps"].notna() & d["ttft"].notna()
    # Decode speed & context: % gap from best (higher value = better).
    d["rank_tps"] = _pct_score_higher_better(d["tps"], valid)
    d["rank_ctx"] = _pct_score_higher_better(d["context_bin"], valid)
    # Peer-rank remaining metrics; rank_ctx / rank_tps already set.
    d = _assign_ranks(d, valid, [
        ("pass_rate_adj", "rank_pass", False),
        ("ttft",          "rank_ttft", True),
        ("gpu_mb",        "rank_gpu",  True),
    ], rank_agg=rank_agg)

    def _ctx_label(b):
        return "<35k" if int(b) == LOW_CONTEXT_BIN else f"{b / 1000:g}k"

    d["label"] = d["series"] + " @" + d["context_bin"].map(_ctx_label)
    return d


def rank_sum_per_model(ranks, rank_agg="sum"):
    """
    Average metric scores across context windows per model, then rank.

    Context capability = max planned context size completed, with fixed ranks
    (180k→1 … 50k→5, <35k→6). Models with NaN decode speed or TTFT go to the bottom.
    """
    cols = ["pass_rate_adj", "tps", "ttft", "gpu_mb"]
    if "context_bin" not in ranks.columns:
        ranks = ranks.copy()
        planned = discover_context_bins(ranks["context"])
        ranks["context_bin"] = ranks["context"].map(
            lambda c: snap_context(c, planned)
        )
    d = ranks.groupby("series", as_index=False).agg(
        **{c: (c, "mean") for c in cols},
        max_context=("context_bin", "max"),
        n_contexts=("context_bin", "nunique"),
    )
    # Keep NaNs so we can detect incomplete models (do not fill with 0).
    d["max_context"] = d["max_context"].fillna(0.0)

    valid = d["tps"].notna() & d["ttft"].notna()
    d["rank_tps"] = _pct_score_higher_better(d["tps"], valid)
    d["rank_ctx"] = _pct_score_higher_better(d["max_context"], valid)
    d = _assign_ranks(d, valid, [
        ("pass_rate_adj", "rank_pass", False),
        ("ttft",          "rank_ttft", True),
        ("gpu_mb",        "rank_gpu",  True),
    ], rank_agg=rank_agg)
    d["label"] = d["series"]
    return d


def _fmt_metric(col, val):
    """Format a real metric value for heatmap cell text."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "n/a"
    if col in ("pass_rate_adj",):
        return f"{val:.0f}%"
    if col in ("tps",):
        return f"{val:.0f}"
    if col in ("ttft",):
        return f"{val:.2f}s"
    if col in ("gpu_mb",):
        return f"{val / 1024:.1f}G" if val >= 1024 else f"{val:.0f}M"
    if col in ("context_bin", "max_context", "context"):
        if int(val) == LOW_CONTEXT_BIN:
            return "<35k"
        return f"{val / 1000:g}k"
    return f"{val:.1f}"


def _draw_rank_sum_figure(r, out_dir, filename, title, value_cols,
                          rank_agg="sum"):
    """Shared 3-panel rank figure. Colour = rank; cell text = real values."""
    if r.empty:
        return

    r = r.sort_values("rank_sum", ascending=True).reset_index(drop=True)
    n = len(r)
    y = np.arange(n)
    y_plot = n - 1 - y  # index 0 (best) at top
    fig_h = max(4.0, 0.45 * n + 1.5)
    agg_label = {"sum": "sum", "mean": "mean", "median": "median"}.get(
        rank_agg, rank_agg)

    fig, (ax_sum, ax_tps, ax_heat) = plt.subplots(
        1, 3, figsize=(17, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1.0, 1.4]},
    )
    for ax in (ax_sum, ax_tps, ax_heat):
        ax.set_ylim(-0.5, n - 0.5)

    labels = list(r["label"])

    # --- panel 1: combined rank score ---
    rs = r["rank_sum"].to_numpy(dtype=float)
    rs_norm = (rs - rs.min()) / max(rs.max() - rs.min(), 1e-9)
    colors_sum = plt.get_cmap("RdYlGn_r")(rs_norm)
    ax_sum.barh(y_plot, rs, color=colors_sum, edgecolor="black", linewidth=0.4,
                height=0.8)
    ax_sum.set_yticks(y_plot)
    ax_sum.set_yticklabels(labels, fontsize=8)
    ax_sum.set_xlabel(f"Rank {agg_label} (lower = better)")
    ax_sum.set_title(f"Overall rank {agg_label}")
    ax_sum.grid(axis="x", alpha=0.3)
    for yi, val in zip(y_plot, rs):
        ax_sum.text(val + 0.15, yi, f"{val:.1f}", va="center", fontsize=7)

    # --- panel 2: tokens/s ---
    tps = r["tps"].to_numpy(dtype=float)
    tps_plot = np.nan_to_num(tps, nan=0.0)
    tps_valid = tps[np.isfinite(tps)]
    if len(tps_valid):
        tps_norm = (tps_plot - tps_valid.min()) / max(
            tps_valid.max() - tps_valid.min(), 1e-9)
        colors_tps = plt.get_cmap("RdYlGn")(tps_norm)
    else:
        colors_tps = "#cccccc"
    ax_tps.barh(y_plot, tps_plot, color=colors_tps, edgecolor="black",
                linewidth=0.4, height=0.8)
    ax_tps.set_xlabel("Decode speed (tok/s)")
    ax_tps.set_title("Tokens / second")
    ax_tps.grid(axis="x", alpha=0.3)
    for yi, val, raw in zip(y_plot, tps_plot, tps):
        txt = "n/a" if not np.isfinite(raw) else f"{val:.0f}"
        ax_tps.text(val + max(float(tps_plot.max()) * 0.01, 0.5), yi, txt,
                    va="center", fontsize=7)

    # --- panel 3: colour = rank, text = real metric values ---
    metric_cols = list(_RANK_COLS)
    metric_labels = ["Pass rate", "Decode speed", "TTFT", "GPU peak", "Context"]
    heat = r[metric_cols].to_numpy(dtype=float)
    heat_draw = heat[::-1]

    mesh = ax_heat.pcolormesh(
        np.arange(len(metric_cols) + 1),
        np.arange(n + 1) - 0.5,
        heat_draw,
        cmap="RdYlGn_r",
        vmin=1,
        vmax=max(float(np.nanmax(heat)), float(n), 1.0),
        edgecolors="white",
        linewidth=0.5,
    )
    ax_heat.set_xticks(np.arange(len(metric_labels)) + 0.5)
    ax_heat.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=8)
    ax_heat.set_title("Metrics (colour = rank, text = value)")
    for i in range(n):
        for j, vcol in enumerate(value_cols):
            raw = r.iloc[i][vcol] if vcol in r.columns else np.nan
            ax_heat.text(j + 0.5, y_plot[i], _fmt_metric(vcol, raw),
                         ha="center", va="center", fontsize=7)

    fig.colorbar(mesh, ax=ax_heat, fraction=0.046, pad=0.04, label="Rank")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, out_dir, filename)


def plot_rank_sum(ranks, out_dir, rank_agg="sum"):
    """Rank figure for each model × context."""
    _draw_rank_sum_figure(
        ranks, out_dir, "09_rank_sum.png",
        f"Model × context ranking (rank {rank_agg}; "
        f"pass/speed ×2, others ×1)\n"
        "(pass rate adjusted for early aborts; colour rank 1 = best)",
        value_cols=_VALUE_COLS_CTX,
        rank_agg=rank_agg,
    )


def plot_rank_sum_per_model(ranks_model, out_dir, rank_agg="sum"):
    """Rank figure per model (metrics averaged across context windows)."""
    _draw_rank_sum_figure(
        ranks_model, out_dir, "10_rank_sum_per_model.png",
        f"Per-model ranking (rank {rank_agg}; pass/speed ×2, others ×1)\n"
        "(incomplete models at bottom; colour rank 1 = best)",
        value_cols=_VALUE_COLS_MODEL,
        rank_agg=rank_agg,
    )


# =============================================================================
# DRIVER
# =============================================================================

def make_plots_for(df, out_dir, rank_agg="sum"):
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = agg_by_context(df)
    lb = leaderboard(df)
    trend = trend_table(agg)
    styles = style_map(sorted(agg["series"].unique()))
    ranks = rank_sum_table(agg, rank_agg=rank_agg)
    ranks_model = rank_sum_per_model(ranks, rank_agg=rank_agg)

    agg.to_csv(out_dir / "by_model_and_context.csv", index=False)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)
    ranks.to_csv(out_dir / "rank_sum.csv", index=False)
    ranks_model.to_csv(out_dir / "rank_sum_per_model.csv", index=False)
    if not trend.empty:
        trend.to_csv(out_dir / "trends.csv", index=False)

    print("\nLeaderboard (pooled over contexts):")
    print(lb.round(1).to_string(index=False))
    if not trend.empty:
        print("\nTrends vs context size:")
        print(trend.round(2).to_string(index=False))

    print("\nRank-sum (model × context, lower = better):")
    show = ranks[["label", "pass_rate_adj", "tps", "ttft", "gpu_mb", "context",
                  "rank_pass", "rank_tps", "rank_ttft", "rank_gpu", "rank_ctx",
                  "rank_sum"]]
    print(show.round(1).to_string(index=False))

    print("\nRank-sum per model (avg across contexts, lower = better):")
    show_m = ranks_model[["label", "pass_rate_adj", "tps", "ttft", "gpu_mb",
                          "max_context", "n_contexts",
                          "rank_pass", "rank_tps", "rank_ttft", "rank_gpu",
                          "rank_ctx", "rank_sum"]]
    print(show_m.round(1).to_string(index=False))

    plot_ranking(lb, out_dir)
    plot_pass_vs_context(agg, styles, out_dir)
    plot_heatmap(agg, out_dir)
    plot_tps_vs_context(agg, styles, out_dir)
    plot_tps_retention(agg, trend, styles, out_dir)
    plot_ttft_vs_context(agg, styles, out_dir)
    plot_tradeoff(agg, styles, out_dir)
    plot_gpu_vs_context(agg, styles, out_dir)
    plot_rank_sum(ranks, out_dir, rank_agg=rank_agg)
    plot_rank_sum_per_model(ranks_model, out_dir, rank_agg=rank_agg)


def make_all(results_file, out_dir, rank_agg="sum", min_context=None):
    results_file, out_dir = Path(results_file), Path(out_dir)
    if not results_file.exists():
        print(f"No results file: {results_file}")
        return

    df = load_results(results_file)
    if df.empty:
        print("Results file is empty; nothing to plot.")
        return

    if min_context is not None:
        before = len(df)
        # Keep runs strictly above the threshold (e.g. --min-context 100000
        # drops 50k/100k and keeps 120k+).
        df = df[df["context_actual"] > min_context].copy()
        print(f"Filtered context_actual > {min_context:,}: "
              f"{before} -> {len(df)} rows")
        if df.empty:
            print("No rows left after context filter; nothing to plot.")
            return

    fills = sorted(float(x) for x in df["fill_ratio"].dropna().unique())
    print(f"\nPlotting {len(df)} results (fill ratios: {fills})"
          f"  rank_agg={rank_agg}"
          + (f"  min_context={min_context}" if min_context is not None else ""))

    for fill in fills:
        sub = df[df["fill_ratio"] == fill]
        target = out_dir / f"fill{int(round(fill * 100))}" if len(fills) > 1 else out_dir
        print(f"\n--- fill ratio {fill} -> {target} ---")
        make_plots_for(sub, target, rank_agg=rank_agg)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--results", default=None,
        help="One results CSV. Default: every per-problem CSV in --folder.")
    parser.add_argument(
        "--folder", default=".",
        help="Folder to search for results CSVs when --results is not given (default: current folder).")
    parser.add_argument("--out", default=None,
                        help="Output folder. Default: a plots/ folder next to each results file.")
    parser.add_argument(
        "--rank-agg", choices=["sum", "mean", "median"], default="sum",
        help="How to combine per-metric ranks in figures 09/10 "
             "(default: sum).")
    parser.add_argument(
        "--min-context", type=int, default=None,
        help="Only include runs with context_actual strictly greater than "
             "this value (e.g. 100000 keeps 120k/150k/180k).")
    args = parser.parse_args()

    try:
        inputs = resolve_inputs(args.results, args.folder)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)

    if not inputs:
        print(f"No per-problem results CSV found in {Path(args.folder).resolve()}")
        print("Expected files like coding_benchmark_results.csv or "
              "coding_benchmark_results_mbpp.csv (not the *_summary.csv files).")
        raise SystemExit(1)

    print(f"Results to plot: {', '.join(p.name for p, _ in inputs)}")
    for path, sub in inputs:
        base = Path(args.out) if args.out else path.parent / "plots"
        out = base / sub if sub else base
        print(f"\n===== {path.name} -> {out} =====")
        try:
            make_all(path, out, rank_agg=args.rank_agg, min_context=args.min_context)
        except Exception as e:
            print(f"Plotting {path.name} failed: {e}")


if __name__ == "__main__":
    main()
