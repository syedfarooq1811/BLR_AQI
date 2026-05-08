from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
JOURNAL_DIR = REPORTS_DIR / "journal_validation"
STREET_DIR = REPORTS_DIR / "street_level"
PAPER_DIR = REPORTS_DIR / "paper_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all paper experiment pipelines and build one consolidated report.")
    parser.add_argument("--quick", action="store_true", help="Run faster but less thorough sweeps where supported.")
    parser.add_argument("--with-downscaler", action="store_true", help="Also train supervised street downscaler if labels exist.")
    parser.add_argument(
        "--downscaler-use-station-proxy-labels",
        action="store_true",
        help="Allow downscaler training with station-derived proxy labels if street labels are unavailable.",
    )
    parser.add_argument("--downscaler-model-zoo", action="store_true", help="Enable strong model zoo + stacking for downscaler.")
    parser.add_argument("--downscaler-paper-novelty-model", action="store_true", help="Enable the named STARLING-AQI paper novelty ensemble.")
    parser.add_argument("--downscaler-max-trials", type=int, default=18, help="Max HGB trials when not using model zoo.")
    parser.add_argument("--downscaler-max-iter", type=int, default=280, help="Max iterations for downscaler learners.")
    parser.add_argument("--downscaler-max-samples", type=int, default=60000, help="Training sample cap for downscaler.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run child scripts.")
    parser.add_argument("--target-r2", type=float, default=0.94, help="Minimum required R2 target.")
    parser.add_argument("--target-rmse", type=float, default=0.15, help="Maximum allowed RMSE target.")
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> dict:
    started = perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = perf_counter() - started
    result = {
        "step": name,
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "ok": proc.returncode == 0,
    }
    return result


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_outputs() -> dict:
    temporal = safe_read_csv(JOURNAL_DIR / "temporal_holdout_metrics.csv")
    loso = safe_read_csv(JOURNAL_DIR / "leave_one_station_out_metrics.csv")
    ablation = safe_read_csv(JOURNAL_DIR / "ablation_metrics.csv")
    interp = safe_read_csv(STREET_DIR / "interpolation_loso_tuning.csv")

    summary: dict[str, object] = {}

    if not temporal.empty:
        best_temporal = temporal.sort_values("rmse").iloc[0].to_dict()
        summary["best_temporal"] = best_temporal
    if not loso.empty:
        loso_model = loso[loso["model"] == "HistGradientBoosting"]
        if not loso_model.empty:
            agg = (
                loso_model.groupby(["model", "feature_set"], as_index=False)
                .agg(rmse=("rmse", "mean"), r2=("r2", "mean"), mae=("mae", "mean"))
                .sort_values("rmse")
            )
            summary["best_loso_model"] = agg.iloc[0].to_dict()
    if not ablation.empty:
        summary["ablation_ranked"] = ablation.sort_values("rmse")[["model", "feature_set", "rmse", "r2", "mae"]].to_dict("records")
    if not interp.empty:
        summary["best_interpolation"] = interp.sort_values("rmse").iloc[0].to_dict()

    downscaler_summary = ROOT / "models" / "street_downscaler" / "street_downscaler_summary.json"
    if downscaler_summary.exists():
        summary["downscaler"] = json.loads(downscaler_summary.read_text(encoding="utf-8"))
    return summary


def evaluate_targets(summary: dict, target_r2: float, target_rmse: float) -> dict:
    checks: dict[str, dict] = {}
    for key in ("best_temporal", "best_loso_model", "best_interpolation"):
        metrics = summary.get(key)
        if not metrics:
            checks[key] = {"available": False, "pass": False, "reason": "missing metrics"}
            continue
        r2 = float(metrics.get("r2", float("-inf")))
        rmse = float(metrics.get("rmse", float("inf")))
        passed = (r2 >= target_r2) and (rmse <= target_rmse)
        checks[key] = {
            "available": True,
            "pass": passed,
            "r2": r2,
            "rmse": rmse,
            "target_r2": target_r2,
            "target_rmse": target_rmse,
        }

    downscaler = summary.get("downscaler", {})
    horizon_checks = []
    for item in downscaler.get("results", []):
        test = item.get("test", {})
        r2 = float(test.get("r2", float("-inf")))
        rmse = float(test.get("rmse", float("inf")))
        horizon_checks.append(
            {
                "horizon_hours": int(item.get("horizon_hours", -1)),
                "pass": (r2 >= target_r2) and (rmse <= target_rmse),
                "r2": r2,
                "rmse": rmse,
                "target_r2": target_r2,
                "target_rmse": target_rmse,
            }
        )
    checks["downscaler_horizons"] = horizon_checks

    overall_pass = all(v.get("pass", False) for k, v in checks.items() if k != "downscaler_horizons")
    if horizon_checks:
        overall_pass = overall_pass and all(item["pass"] for item in horizon_checks)
    checks["overall_pass"] = overall_pass
    return checks


def write_report(step_results: list[dict], summary: dict, target_checks: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "pipeline_steps.json").write_text(json.dumps(step_results, indent=2), encoding="utf-8")
    (PAPER_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PAPER_DIR / "target_checks.json").write_text(json.dumps(target_checks, indent=2), encoding="utf-8")

    lines = [
        "# Paper Experiment Harness Report",
        "",
        "This report is generated by `scripts/run_paper_experiments.py`.",
        "",
        "## Pipeline Steps",
        "",
    ]
    for step in step_results:
        status = "OK" if step["ok"] else "FAILED"
        lines += [
            f"- `{step['step']}`: **{status}** in {step['elapsed_seconds']}s",
            f"  - Command: `{step['command']}`",
        ]
        if not step["ok"] and step["stderr"]:
            lines.append(f"  - Error: `{step['stderr'].splitlines()[-1]}`")

    lines += ["", "## Consolidated Metrics", ""]
    if "best_temporal" in summary:
        bt = summary["best_temporal"]
        lines += [
            "### Best Temporal Holdout",
            f"- Model: `{bt.get('model')}` / `{bt.get('feature_set')}`",
            f"- RMSE: `{bt.get('rmse'):.4f}`",
            f"- R2: `{bt.get('r2'):.4f}`",
            f"- MAE: `{bt.get('mae'):.4f}`",
            "",
        ]
    if "best_loso_model" in summary:
        bl = summary["best_loso_model"]
        lines += [
            "### Best LOSO Model",
            f"- Model: `{bl.get('model')}` / `{bl.get('feature_set')}`",
            f"- RMSE: `{bl.get('rmse'):.4f}`",
            f"- R2: `{bl.get('r2'):.4f}`",
            f"- MAE: `{bl.get('mae'):.4f}`",
            "",
        ]
    if "best_interpolation" in summary:
        bi = summary["best_interpolation"]
        lines += [
            "### Best Station-to-Street Interpolation",
            f"- IDW power: `{bi.get('idw_power')}`",
            f"- IDW blend: `{bi.get('idw_blend')}`",
            f"- RMSE: `{bi.get('rmse'):.4f}`",
            f"- R2: `{bi.get('r2'):.4f}`",
            "",
        ]
    if "downscaler" in summary:
        lines += [
            "### Supervised Street Downscaler",
            "- Found `models/street_downscaler/street_downscaler_summary.json`.",
            "- See JSON summary for 24h and 168h horizon metrics.",
            "",
        ]
    lines += ["## Target Gates", ""]
    lines += [f"- Overall pass: **{target_checks.get('overall_pass', False)}**", ""]
    for key, info in target_checks.items():
        if key in {"overall_pass", "downscaler_horizons"}:
            continue
        lines.append(f"- `{key}`: pass={info.get('pass')} r2={info.get('r2')} rmse={info.get('rmse')}")
    if target_checks.get("downscaler_horizons"):
        lines.append("- `downscaler_horizons`:")
        for item in target_checks["downscaler_horizons"]:
            lines.append(
                f"  - h{item['horizon_hours']}: pass={item['pass']} r2={item['r2']:.4f} rmse={item['rmse']:.4f}"
            )
    lines.append("")
    lines += [
        "## Notes",
        "- This harness centralizes temporal validation, LOSO, ablations, and interpolation tuning.",
        "- Supervised downscaler metrics require real street labels in `data/raw/street_labels.parquet`.",
    ]

    (PAPER_DIR / "paper_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = perf_counter()
    step_results: list[dict] = []

    cmd_journal = [args.python, "scripts/run_journal_validation.py"]
    if args.quick:
        cmd_journal.append("--quick")
    step_results.append(run_step("journal_validation", cmd_journal))

    cmd_interp = [args.python, "scripts/validate_spatial_interpolation.py"]
    step_results.append(run_step("spatial_interpolation_loso", cmd_interp))

    if args.with_downscaler:
        cmd_down = [
            args.python,
            "scripts/train_street_downscaler.py",
            "--max-trials",
            str(args.downscaler_max_trials),
            "--max-iter",
            str(args.downscaler_max_iter),
            "--max-samples",
            str(args.downscaler_max_samples),
            "--target-r2",
            str(args.target_r2),
            "--target-rmse",
            str(args.target_rmse),
        ]
        if args.downscaler_use_station_proxy_labels:
            cmd_down.append("--use-station-proxy-labels")
        if args.downscaler_model_zoo:
            cmd_down.append("--model-zoo")
        if args.downscaler_paper_novelty_model:
            cmd_down.append("--paper-novelty-model")
        step_results.append(run_step("street_downscaler_supervised", cmd_down))

    summary = summarize_outputs()
    target_checks = evaluate_targets(summary, target_r2=args.target_r2, target_rmse=args.target_rmse)
    summary["total_elapsed_seconds"] = round(perf_counter() - started, 3)
    write_report(step_results, summary, target_checks)

    print(f"Wrote consolidated outputs to {PAPER_DIR}")
    failed = [s for s in step_results if not s["ok"]]
    if failed:
        print(f"{len(failed)} step(s) failed. See pipeline_steps.json for details.")
        raise SystemExit(1)
    if not target_checks.get("overall_pass", False):
        print("Target checks failed. See target_checks.json for exact gaps.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
