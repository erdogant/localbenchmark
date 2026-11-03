"""
LocalBenchmark UI — clean Streamlit front-end for exploring results and
launching new runs when a new model arrives.

Run from the package parent (or any cwd that can import localbenchmark):

    streamlit run localbenchmark/streamlit_app.py

Requires: streamlit, plotly, pandas (and the existing localbenchmark deps).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & persistence
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "ui_config.json"
KNOWN_RESULT_GLOBS = [
    "**/coding_benchmark_results.csv",
    "**/mbpp_results.csv",
    "**/swebench_results.csv",
    "**/suite_results_*.csv",
    "**/dryrun_estimates.csv",
    "**/dryrun_leaderboard.csv",
    "**/*results*.csv",
]
CORE_COLS = [
    "model", "context_requested", "context_actual", "cache_label",
    "passed", "decode_tokens_per_sec", "ttft", "gpu_memory_mb",
    "prompt_tokens", "completion_tokens", "experiment_id", "problem_id",
    "score", "error_type", "item_type", "domain",
]
# Plots produced by benchmark_plots.make_plots_for / make_all
FEATURED_PLOTS = [
    ("04_tps_vs_context.png", "TPS vs context"),
    ("07_score_vs_speed.png", "Score vs speed"),
    ("09_rank_sum.png", "Rank sum"),
    ("10_rank_sum_per_model.png", "Rank sum per model"),
]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    defaults = {
        "output_root": str(Path.cwd()),
        "base_url": "http://localhost:1234",
        "backend": "auto",
        "api_key": "",
        "python_executable": sys.executable,
        "default_contexts": [50000, 80000, 100000, 120000, 150000, 180000],
        "default_benchmark": "humaneval",
        "last_models": [],
        "last_results": [],
    }
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in data.items() if k in defaults})
        except Exception:
            pass
    return defaults


def save_config(cfg: dict[str, Any], path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def discover_result_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in KNOWN_RESULT_GLOBS:
        for p in root.glob(pattern):
            key = str(p.resolve())
            if key not in seen and p.is_file() and p.stat().st_size > 0:
                seen.add(key)
                found.append(p)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    for col in ("passed",):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "yes"))
    for col in ("score", "decode_tokens_per_sec", "ttft", "gpu_memory_mb",
                "context_requested", "context_actual", "prompt_tokens",
                "completion_tokens"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_source"] = str(path)
    df["_source_name"] = path.name
    return df


def find_plot_images(results_path: Path) -> list[tuple[str, Path]]:
    """Locate featured plot PNGs next to a results CSV (plots/ or fill* subdirs)."""
    parent = results_path.parent
    search_dirs = [parent / "plots", parent]
    search_dirs.extend(sorted(parent.glob("fill*")))
    search_dirs.extend(sorted((parent / "plots").glob("fill*")) if (parent / "plots").is_dir() else [])
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for filename, label in FEATURED_PLOTS:
        for d in search_dirs:
            candidate = d / filename
            if candidate.is_file() and filename not in seen:
                found.append((label, candidate))
                seen.add(filename)
                break
    return found


def show_featured_plots(results_path: Path) -> None:
    plots = find_plot_images(results_path)
    if not plots:
        st.caption(
            f"No featured plots found near `{results_path.name}` "
            "(expected under a sibling `plots/` folder: "
            "04_tps_vs_context, 07_score_vs_speed, 09_rank_sum, 10_rank_sum_per_model)."
        )
        return
    st.subheader("Benchmark plots")
    # Two per row
    for i in range(0, len(plots), 2):
        cols = st.columns(2)
        for col, (label, path) in zip(cols, plots[i : i + 2]):
            with col:
                st.markdown(f"**{label}**")
                st.image(str(path), use_container_width=True)
                st.caption(path.name)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model × context (actual preferred)."""
    if df.empty:
        return pd.DataFrame()
    ctx = "context_actual" if "context_actual" in df.columns else "context_requested"
    group_cols = ["model"]
    if ctx in df.columns:
        group_cols.append(ctx)
    if "cache_label" in df.columns:
        group_cols.append("cache_label")

    # Prefer scored suites when score is present
    has_score = "score" in df.columns and df["score"].notna().any()
    aggs: dict[str, Any] = {"n": ("problem_id", "count") if "problem_id" in df.columns else ("model", "size")}
    if "passed" in df.columns:
        aggs["pass_rate"] = ("passed", lambda s: 100.0 * s.mean())
    if has_score:
        aggs["mean_score"] = ("score", "mean")
    if "decode_tokens_per_sec" in df.columns:
        aggs["mean_tps"] = ("decode_tokens_per_sec", "mean")
    if "ttft" in df.columns:
        aggs["mean_ttft"] = ("ttft", "mean")
    if "gpu_memory_mb" in df.columns:
        aggs["mean_gpu_mb"] = ("gpu_memory_mb", "mean")

    out = df.groupby(group_cols, dropna=False).agg(**aggs).reset_index()
    if ctx in out.columns:
        out = out.rename(columns={ctx: "context"})
    return out.sort_values(
        by=[c for c in ("pass_rate", "mean_score", "mean_tps") if c in out.columns],
        ascending=False,
    )


def server_models(base_url: str) -> list[str]:
    """Best-effort list of models from an OpenAI-compatible or LM Studio server."""
    try:
        import requests
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            candidates = [f"{url}/v1/models", f"{url}/api/v0/models", f"{url}/models"]
        else:
            candidates = [f"{url}/models"]
        for u in candidates:
            r = requests.get(u, timeout=3)
            if r.ok:
                data = r.json()
                items = data.get("data") or data.get("models") or []
                names = []
                for m in items:
                    if isinstance(m, dict):
                        names.append(m.get("id") or m.get("model") or m.get("name"))
                    else:
                        names.append(str(m))
                return [n for n in names if n]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# UI pages
# ---------------------------------------------------------------------------

def check_python_env(python_exe: str) -> tuple[bool, str]:
    """Return (ok, message) after probing the chosen interpreter for required packages."""
    probe = (
        "import importlib.util, sys\n"
        "missing = [m for m in ('openai', 'pandas', 'requests') if importlib.util.find_spec(m) is None]\n"
        "print(sys.executable)\n"
        "print('OK' if not missing else 'MISSING:' + ','.join(missing))\n"
    )
    try:
        r = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (r.stdout or "").strip().splitlines()
        if r.returncode != 0:
            return False, (r.stderr or r.stdout or "probe failed").strip()
        exe_line = out[0] if out else python_exe
        status = out[1] if len(out) > 1 else ""
        if status.startswith("MISSING:"):
            return False, f"{exe_line} is missing packages: {status[8:]}"
        return True, exe_line
    except Exception as e:
        return False, str(e)


def page_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    st.header("Settings")
    st.caption("Values are saved to `ui_config.json` and restored on next launch.")

    c1, c2 = st.columns(2)
    with c1:
        cfg["output_root"] = st.text_input(
            "Output root folder",
            value=cfg.get("output_root", str(Path.cwd())),
            help="Root directory where benchmark subfolders (humaneval/, mbpp/, suite/, …) are written and discovered.",
        )
        cfg["base_url"] = st.text_input("Server base URL", value=cfg.get("base_url", "http://localhost:1234"))
        cfg["backend"] = st.selectbox(
            "Backend",
            options=["auto", "lmstudio", "openai"],
            index=["auto", "lmstudio", "openai"].index(cfg.get("backend", "auto")),
        )
        cfg["python_executable"] = st.text_input(
            "Python executable for runs",
            value=cfg.get("python_executable") or sys.executable,
            help="Use the same Python that works from the command line (the one that has openai installed).",
        )
    with c2:
        cfg["api_key"] = st.text_input("API key (optional)", value=cfg.get("api_key", ""), type="password")
        ctx_text = st.text_input(
            "Default contexts (comma-separated)",
            value=", ".join(str(c) for c in cfg.get("default_contexts", [])),
        )
        try:
            cfg["default_contexts"] = sorted({int(x.strip()) for x in ctx_text.split(",") if x.strip()})
        except ValueError:
            st.warning("Contexts must be integers.")
        cfg["default_benchmark"] = st.selectbox(
            "Default benchmark",
            options=["humaneval", "mbpp", "swebench", "suite"],
            index=["humaneval", "mbpp", "swebench", "suite"].index(cfg.get("default_benchmark", "humaneval")),
        )

    ok, msg = check_python_env(cfg["python_executable"])
    if ok:
        st.success(f"Python OK: `{msg}` (openai / pandas / requests present)")
    else:
        st.error(
            f"Python check failed: {msg}\n\n"
            "Set **Python executable for runs** to the interpreter you use on the command line "
            "(e.g. the path shown by `where python` / `which python` in the working env)."
        )

    if st.button("Save settings", type="primary"):
        save_config(cfg)
        st.success(f"Saved to {DEFAULT_CONFIG_PATH}")
    return cfg


def page_compare(cfg: dict[str, Any]) -> None:
    st.header("Quick comparison")
    st.caption("Compare models across contexts from one or more result files.")

    root = Path(cfg["output_root"]).expanduser()
    discovered = discover_result_files(root)

    options = [str(p) for p in discovered]
    default_sel = [o for o in cfg.get("last_results", []) if o in options] or options[:3]

    selected = st.multiselect(
        "Result files",
        options=options,
        default=default_sel,
        help="Auto-discovered under the output root. Pick several to compare across runs.",
    )
    manual = st.text_input("Or paste an extra CSV path")
    if manual.strip():
        selected = list(selected) + [manual.strip()]

    if not selected:
        st.info("No result files selected. Run a benchmark or point the output root at an existing results folder.")
        return

    frames = []
    for s in selected:
        p = Path(s)
        if p.exists():
            try:
                frames.append(load_results_csv(p))
            except Exception as e:
                st.warning(f"Could not load {p.name}: {e}")
    if not frames:
        st.warning("None of the selected files could be loaded.")
        return

    df = pd.concat(frames, ignore_index=True)
    cfg["last_results"] = selected
    save_config(cfg)

    summary = summarise(df)
    if summary.empty:
        st.warning("No aggregatable rows found.")
        return

    st.subheader("Summary table")
    st.dataframe(
        summary.style.format({
            c: "{:.1f}" for c in summary.columns
            if c in ("pass_rate", "mean_score", "mean_tps", "mean_ttft", "mean_gpu_mb")
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    # Charts
    metric_candidates = [c for c in ("pass_rate", "mean_score", "mean_tps", "mean_ttft") if c in summary.columns]
    if metric_candidates:
        metric = st.selectbox("Chart metric", metric_candidates, index=0)
        if "context" in summary.columns:
            fig = px.bar(
                summary,
                x="context",
                y=metric,
                color="model",
                barmode="group",
                title=f"{metric} by context",
                labels={"context": "Context tokens", metric: metric.replace("_", " ").title()},
            )
            fig.update_layout(template="plotly_white", legend_title_text="Model", height=420)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(
                summary.sort_values("context"),
                x="context",
                y=metric,
                color="model",
                markers=True,
                title=f"{metric} trend",
            )
            fig2.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = px.bar(summary, x="model", y=metric, title=metric)
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Saved matplotlib plots from benchmark_plots (per selected file)
    st.divider()
    for s in selected:
        p = Path(s)
        if not p.exists():
            continue
        with st.expander(f"Plots for `{p.name}`", expanded=(s == selected[0])):
            show_featured_plots(p)


def page_explore(cfg: dict[str, Any]) -> None:
    st.header("Explore results")
    root = Path(cfg["output_root"]).expanduser()
    discovered = discover_result_files(root)
    if not discovered:
        st.info("No result CSVs found under the output root.")
        return

    choice = st.selectbox("Result file", options=[str(p) for p in discovered])
    df = load_results_csv(Path(choice))

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        models = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
        sel_models = st.multiselect("Models", models, default=models)
    with f2:
        if "context_actual" in df.columns:
            ctxs = sorted(df["context_actual"].dropna().unique())
            sel_ctx = st.multiselect("Contexts", ctxs, default=ctxs)
        else:
            sel_ctx = None
    with f3:
        if "cache_label" in df.columns:
            labels = sorted(df["cache_label"].dropna().astype(str).unique())
            sel_lbl = st.multiselect("Cache label", labels, default=labels)
        else:
            sel_lbl = None

    view = df.copy()
    if sel_models and "model" in view.columns:
        view = view[view["model"].isin(sel_models)]
    if sel_ctx is not None and "context_actual" in view.columns:
        view = view[view["context_actual"].isin(sel_ctx)]
    if sel_lbl is not None and "cache_label" in view.columns:
        view = view[view["cache_label"].astype(str).isin(sel_lbl)]

    st.caption(f"{len(view):,} rows after filters")
    show_cols = [c for c in CORE_COLS if c in view.columns]
    st.dataframe(view[show_cols] if show_cols else view, use_container_width=True, height=420)

    # Quick distribution
    if "passed" in view.columns and view["passed"].notna().any():
        by_model = view.groupby("model")["passed"].mean().mul(100).reset_index(name="pass_rate")
        fig = px.bar(by_model, x="model", y="pass_rate", title="Pass rate % (filtered)")
        fig.update_layout(template="plotly_white", height=360, yaxis_title="Pass rate %")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    show_featured_plots(Path(choice))


def page_run(cfg: dict[str, Any]) -> None:
    st.header("Run a benchmark")
    st.caption("Configure a new run (or dry-run) when a new model arrives. Output goes under the folder you set.")

    root = Path(cfg["output_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    bench = st.selectbox(
        "Benchmark",
        options=["humaneval", "mbpp", "swebench", "suite"],
        index=["humaneval", "mbpp", "swebench", "suite"].index(cfg.get("default_benchmark", "humaneval")),
    )

    # Models
    live = server_models(cfg["base_url"])
    if live:
        st.caption(f"Server at {cfg['base_url']} reports {len(live)} model(s).")
        models = st.multiselect("Models", options=live, default=[m for m in cfg.get("last_models", []) if m in live] or live[:1])
    else:
        models_text = st.text_area(
            "Models (one per line or space-separated)",
            value="\n".join(cfg.get("last_models", [])),
            height=100,
        )
        models = [m.strip() for m in models_text.replace(",", " ").split() if m.strip()]

    ctx_default = cfg.get("default_contexts", [50000, 100000, 150000])
    contexts = st.multiselect(
        "Contexts",
        options=sorted(set(ctx_default + [50000, 80000, 100000, 120000, 150000, 180000])),
        default=ctx_default,
    )

    c1, c2 = st.columns(2)
    with c1:
        limit = st.number_input("Limit problems (0 = all)", min_value=0, value=0, step=1)
        dry_run = st.checkbox("Dry-run only (estimates, no model load)", value=False)
    with c2:
        out_folder = st.text_input("Output root for this run", value=str(root))
        results_name = st.text_input("Results CSV name (optional)", value="")

    suite_path = docs_folder = judge = None
    closed_book = False
    if bench == "suite":
        suite_path = st.text_input("Suite file (.jsonl)", value=str(APP_DIR / "tests" / "suites" / "northwind_demo.jsonl"))
        docs_folder = st.text_input("Documents folder", value=str(APP_DIR / "tests" / "company_docs"))
        judge = st.text_input("Judge model (optional)", value="")
        closed_book = st.checkbox("Closed-book (no documents in prompt)", value=False)

    # Build command — use the configured Python (same env as your working CLI)
    python_exe = cfg.get("python_executable") or sys.executable
    script = APP_DIR / "localbenchmark.py"
    cmd = [python_exe, str(script)]
    cmd += ["--benchmark", bench]
    if models:
        cmd += ["--models", *models]
    if contexts:
        cmd += ["--contexts", *[str(c) for c in contexts]]
    cmd += ["--base-url", cfg["base_url"], "--backend", cfg["backend"]]
    if cfg.get("api_key"):
        cmd += ["--api-key", cfg["api_key"]]
    if limit and limit > 0:
        cmd += ["--limit", str(int(limit))]
    if dry_run:
        cmd += ["--dry-run"]
    if results_name.strip():
        cmd += ["--results", results_name.strip()]
    if bench == "suite":
        if suite_path:
            cmd += ["--suite", suite_path]
        if docs_folder:
            cmd += ["--docs", docs_folder]
        if judge:
            cmd += ["--judge", judge]
        if closed_book:
            cmd += ["--closed-book"]

    st.markdown(f"**Python for this run:** `{python_exe}`")
    ok, msg = check_python_env(python_exe)
    if not ok:
        st.warning(
            f"This Python may be missing packages ({msg}). "
            "Set the correct path under **Settings → Python executable for runs** "
            "(use the same interpreter that works from your command line)."
        )

    st.code(" ".join(cmd), language="bash")

    col_a, col_b = st.columns(2)
    launch = col_a.button("Launch run", type="primary", disabled=not models and not dry_run)
    save_models = col_b.button("Remember models & settings")

    if save_models:
        cfg["last_models"] = models
        cfg["default_benchmark"] = bench
        cfg["default_contexts"] = contexts
        cfg["output_root"] = out_folder
        save_config(cfg)
        st.success("Settings updated.")

    if launch:
        cfg["last_models"] = models
        cfg["output_root"] = out_folder
        save_config(cfg)
        workdir = Path(out_folder).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        st.info(f"Working directory: `{workdir}`")
        log_box = st.empty()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ},
            )
            lines: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-80:]), language="text")
            proc.wait()
            if proc.returncode == 0:
                st.success("Run finished successfully.")
            else:
                st.error(f"Process exited with code {proc.returncode}")
                if any("ModuleNotFoundError" in ln for ln in lines):
                    st.error(
                        "Missing package in the Python used for the run. "
                        "Open **Settings**, set **Python executable for runs** to the "
                        "interpreter that works from the command line, save, and retry."
                    )
        except Exception as e:
            st.error(f"Failed to start: {e}")


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="LocalBenchmark",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Light professional chrome
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
        div[data-testid="stSidebar"] { background: #f8fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "cfg" not in st.session_state:
        st.session_state.cfg = load_config()
    cfg = st.session_state.cfg

    with st.sidebar:
        st.title("LocalBenchmark")
        st.caption("Compare · Explore · Run")
        page = st.radio(
            "Navigate",
            options=["Compare", "Explore", "Run benchmark", "Settings"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(f"**Output root**\n\n`{cfg.get('output_root', '.')}`")
        st.markdown(f"**Server**\n\n`{cfg.get('base_url', '')}`")
        if st.button("Reload config from disk"):
            st.session_state.cfg = load_config()
            st.rerun()

    if page == "Compare":
        page_compare(cfg)
    elif page == "Explore":
        page_explore(cfg)
    elif page == "Run benchmark":
        page_run(cfg)
    else:
        st.session_state.cfg = page_settings(cfg)


def cli() -> None:
    """Console entry point: `localbenchmark_app` → streamlit run this file (localhost only)."""
    app_path = str(Path(__file__).resolve())
    # Prefer the Streamlit CLI API so we stay in-process when possible.
    try:
        from streamlit.web import cli as stcli
    except ImportError as e:
        raise SystemExit(
            "streamlit is not installed in this environment. "
            "Install it with: pip install streamlit"
        ) from e
    # Bind to localhost only; do not open a browser automatically.
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        *sys.argv[1:],
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
