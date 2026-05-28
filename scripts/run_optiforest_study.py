#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
import urllib.request
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / "upstream_optiforest"
DATA_DIR = ROOT / "data" / "benchmarks"
RESULTS_DIR = ROOT / "results"

UPSTREAM_REPO = "https://github.com/xiagll/optiforest.git"
UPSTREAM_AD_URL = "https://raw.githubusercontent.com/xiagll/optiforest/main/data/ad.csv"
ADBENCH_BASE_URL = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"
)


@dataclass(frozen=True)
class PaperStats:
    n_samples: int
    n_features: int
    anomaly_rate_pct: float


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    cache_name: str
    paper: PaperStats
    source: str
    note: str = ""
    remote_name: str | None = None


PAPER_DATASETS: OrderedDict[str, DatasetSpec] = OrderedDict(
    [
        (
            "ad",
            DatasetSpec(
                kind="csv",
                cache_name="ad.csv",
                paper=PaperStats(3279, 1555, 13.79),
                source="upstream_optiforest",
                note="Exact sample file from the upstream OptIForest repository.",
            ),
        ),
        (
            "campaign",
            DatasetSpec(
                kind="npz",
                cache_name="5_campaign.npz",
                remote_name="5_campaign.npz",
                paper=PaperStats(41188, 62, 11.27),
                source="adbench",
            ),
        ),
        (
            "arrhythmia",
            DatasetSpec(
                kind="openml_arrhythmia",
                cache_name="arrhythmia_openml.npz",
                paper=PaperStats(452, 274, 14.60),
                source="openml",
                note=(
                    "Built from OpenML Arrhythmia by dropping the five columns that contain "
                    "missing values, which recovers the paper's 274-feature shape."
                ),
            ),
        ),
        (
            "cardio",
            DatasetSpec(
                kind="npz",
                cache_name="6_cardio.npz",
                remote_name="6_cardio.npz",
                paper=PaperStats(1831, 21, 9.61),
                source="adbench",
            ),
        ),
        (
            "backdoor",
            DatasetSpec(
                kind="npz",
                cache_name="3_backdoor.npz",
                remote_name="3_backdoor.npz",
                paper=PaperStats(95329, 196, 2.44),
                source="adbench",
            ),
        ),
        (
            "kddcup99",
            DatasetSpec(
                kind="npz",
                cache_name="16_http.npz",
                remote_name="16_http.npz",
                paper=PaperStats(494021, 38, 1.77),
                source="adbench_http_proxy",
                note=(
                    "The paper cites the full KDDCup99 row, but the released OptIForest code "
                    "does not include a dedicated KDDCup99 loader. This harness uses the "
                    "paper's separate HTTP subset dataset instead for runnable reproduction."
                ),
            ),
        ),
        (
            "celeba",
            DatasetSpec(
                kind="npz",
                cache_name="8_celeba.npz",
                remote_name="8_celeba.npz",
                paper=PaperStats(202599, 39, 2.24),
                source="adbench",
            ),
        ),
        (
            "mnist",
            DatasetSpec(
                kind="npz",
                cache_name="24_mnist.npz",
                remote_name="24_mnist.npz",
                paper=PaperStats(7603, 100, 9.21),
                source="adbench",
            ),
        ),
        (
            "census",
            DatasetSpec(
                kind="npz",
                cache_name="9_census.npz",
                remote_name="9_census.npz",
                paper=PaperStats(299285, 500, 6.20),
                source="adbench",
            ),
        ),
        (
            "donors",
            DatasetSpec(
                kind="npz",
                cache_name="11_donors.npz",
                remote_name="11_donors.npz",
                paper=PaperStats(619326, 10, 5.92),
                source="adbench",
            ),
        ),
        (
            "cover",
            DatasetSpec(
                kind="npz",
                cache_name="10_cover.npz",
                remote_name="10_cover.npz",
                paper=PaperStats(286048, 10, 0.96),
                source="adbench",
            ),
        ),
        (
            "http",
            DatasetSpec(
                kind="npz",
                cache_name="16_http.npz",
                remote_name="16_http.npz",
                paper=PaperStats(567498, 3, 0.39),
                source="adbench",
            ),
        ),
        (
            "smtp",
            DatasetSpec(
                kind="npz",
                cache_name="34_smtp.npz",
                remote_name="34_smtp.npz",
                paper=PaperStats(95156, 3, 0.03),
                source="adbench",
            ),
        ),
        (
            "ionosphere",
            DatasetSpec(
                kind="openml_ionosphere",
                cache_name="ionosphere_openml.npz",
                paper=PaperStats(351, 34, 35.90),
                source="openml",
                note="Uses the 34-feature OpenML Ionosphere dataset to match the paper's shape.",
            ),
        ),
        (
            "satellite",
            DatasetSpec(
                kind="npz",
                cache_name="30_satellite.npz",
                remote_name="30_satellite.npz",
                paper=PaperStats(6435, 36, 31.60),
                source="adbench",
            ),
        ),
        (
            "shuttle",
            DatasetSpec(
                kind="npz",
                cache_name="32_shuttle.npz",
                remote_name="32_shuttle.npz",
                paper=PaperStats(49097, 9, 7.15),
                source="adbench",
            ),
        ),
        (
            "spam",
            DatasetSpec(
                kind="npz",
                cache_name="35_SpamBase.npz",
                remote_name="35_SpamBase.npz",
                paper=PaperStats(4207, 57, 39.91),
                source="adbench",
            ),
        ),
        (
            "vowel",
            DatasetSpec(
                kind="npz",
                cache_name="40_vowels.npz",
                remote_name="40_vowels.npz",
                paper=PaperStats(1456, 12, 3.43),
                source="adbench",
            ),
        ),
        (
            "waveform",
            DatasetSpec(
                kind="npz",
                cache_name="41_Waveform.npz",
                remote_name="41_Waveform.npz",
                paper=PaperStats(3505, 21, 4.62),
                source="adbench",
                note=(
                    "ADBench's released Waveform file is 3443x21 with 2.90% anomalies, so the "
                    "shape diverges from the appendix table."
                ),
            ),
        ),
        (
            "wine",
            DatasetSpec(
                kind="npz",
                cache_name="45_wine.npz",
                remote_name="45_wine.npz",
                paper=PaperStats(5318, 11, 4.53),
                source="adbench",
                note=(
                    "ADBench's released wine dataset is the classic 129x13 ODDS benchmark, not "
                    "the 5318x11 dataset listed in the paper appendix."
                ),
            ),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OptIForest benchmark with the paper's settings."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument("--runs", type=int, default=15, help="Number of repeated runs.")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees.")
    parser.add_argument(
        "--branch",
        type=int,
        default=0,
        help="Branch parameter passed to OptIForest. Use 0 for the paper's e-approximation.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("paper", "fixed"),
        default="paper",
        help="Threshold selection strategy.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Explicit threshold to use with --threshold-mode=fixed.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "optiforest_summary.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--details-output",
        default=str(RESULTS_DIR / "optiforest_runs.csv"),
        help="Per-run CSV output path.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base random seed; each run uses seed_base + run_index.",
    )
    parser.add_argument(
        "--force-refresh-data",
        action="store_true",
        help="Re-download cached dataset files.",
    )
    return parser.parse_args()


def ensure_upstream_repo() -> None:
    if UPSTREAM_DIR.exists():
        return
    subprocess.run(["git", "clone", UPSTREAM_REPO, str(UPSTREAM_DIR)], check=True)


def import_optiforest():
    ensure_upstream_repo()
    sys.path.insert(0, str(UPSTREAM_DIR))
    from detectors import OptIForest  # pylint: disable=import-error

    return OptIForest


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, target: Path, force: bool = False) -> Path:
    if target.exists() and not force:
        return target
    ensure_parent(target)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        handle.write(response.read())
    return target


def prepare_arrhythmia(cache_path: Path, force: bool = False) -> Path:
    if cache_path.exists() and not force:
        return cache_path
    X, y = fetch_openml(
        name="arrhythmia",
        version=1,
        as_frame=False,
        parser="auto",
        return_X_y=True,
    )
    X = X.astype(float)
    X = X[:, ~np.isnan(X).any(axis=0)]
    anomaly_classes = {"3", "4", "5", "7", "8", "9", "14", "15"}
    y = np.isin(y.astype(str), sorted(anomaly_classes)).astype(int)
    ensure_parent(cache_path)
    np.savez_compressed(cache_path, X=X, y=y)
    return cache_path


def prepare_ionosphere(cache_path: Path, force: bool = False) -> Path:
    if cache_path.exists() and not force:
        return cache_path
    X, y = fetch_openml(
        name="ionosphere",
        version=1,
        as_frame=False,
        parser="auto",
        return_X_y=True,
    )
    X = X.astype(float)
    y = (y.astype(str) == "b").astype(int)
    ensure_parent(cache_path)
    np.savez_compressed(cache_path, X=X, y=y)
    return cache_path


GENERATORS: dict[str, Callable[[Path, bool], Path]] = {
    "openml_arrhythmia": prepare_arrhythmia,
    "openml_ionosphere": prepare_ionosphere,
}


def ensure_dataset_file(name: str, spec: DatasetSpec, force: bool = False) -> Path:
    cache_path = DATA_DIR / spec.cache_name
    if spec.kind == "csv":
        return download(UPSTREAM_AD_URL, cache_path, force=force)
    if spec.kind == "npz":
        assert spec.remote_name is not None
        url = f"{ADBENCH_BASE_URL}/{spec.remote_name}"
        return download(url, cache_path, force=force)
    if spec.kind in GENERATORS:
        return GENERATORS[spec.kind](cache_path, force=force)
    raise ValueError(f"Unsupported dataset kind for {name}: {spec.kind}")


def load_dataset(name: str, spec: DatasetSpec, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    path = ensure_dataset_file(name, spec, force=force)
    if spec.kind == "csv":
        df = pd.read_csv(path, header=None)
        X = df.values[:, :-1].astype(float)
        y = df.values[:, -1].astype(float).astype(int)
        return X, y
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(float)
    y = data["y"].astype(int)
    return X, y


def select_threshold(dataset: str, X: np.ndarray, args: argparse.Namespace) -> int:
    if args.threshold_mode == "fixed":
        if args.threshold is None:
            raise ValueError("--threshold is required with --threshold-mode=fixed")
        return args.threshold
    if dataset in {"ad", "vowel"}:
        return 512
    if X.shape[0] > 10_000 or X.shape[1] > 1_000:
        return 403
    return 55


def shape_matches_paper(X: np.ndarray, y: np.ndarray, spec: DatasetSpec) -> bool:
    rate = round(100.0 * float(y.mean()), 2)
    return (
        X.shape[0] == spec.paper.n_samples
        and X.shape[1] == spec.paper.n_features
        and abs(rate - spec.paper.anomaly_rate_pct) <= 0.05
    )


def dataset_list(arg: str) -> list[str]:
    if arg == "all":
        return list(PAPER_DATASETS.keys())
    names = [item.strip().lower() for item in arg.split(",") if item.strip()]
    unknown = [name for name in names if name not in PAPER_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return names


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    OptIForest = import_optiforest()

    selected = dataset_list(args.datasets)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    print(f"Running OptIForest on {len(selected)} dataset(s): {', '.join(selected)}")

    for dataset in selected:
        spec = PAPER_DATASETS[dataset]
        X, y = load_dataset(dataset, spec, force=args.force_refresh_data)
        threshold = select_threshold(dataset, X, args)
        matches = shape_matches_paper(X, y, spec)
        observed_rate = round(100.0 * float(y.mean()), 2)

        aucs: list[float] = []
        prs: list[float] = []
        fit_times: list[float] = []
        pred_times: list[float] = []

        print(
            f"[{dataset}] shape={X.shape} anomalies={int(y.sum())}/{len(y)} "
            f"rate={observed_rate:.2f}% threshold={threshold} match_paper={matches}"
        )

        for run_idx in range(args.runs):
            seed = args.seed_base + run_idx
            np.random.seed(seed)
            random.seed(seed)

            model = OptIForest("L2OPT", args.trees, threshold, args.branch)

            fit_start = time.perf_counter()
            model.fit(X)
            fit_seconds = time.perf_counter() - fit_start

            pred_start = time.perf_counter()
            scores = model.decision_function(X)
            pred_seconds = time.perf_counter() - pred_start

            auc = roc_auc_score(y, -1.0 * scores)
            pr = average_precision_score(y, -1.0 * scores)

            aucs.append(auc)
            prs.append(pr)
            fit_times.append(fit_seconds)
            pred_times.append(pred_seconds)

            detail_rows.append(
                {
                    "dataset": dataset,
                    "run": run_idx + 1,
                    "seed": seed,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    "anomaly_rate_pct": observed_rate,
                    "threshold": threshold,
                    "branch": args.branch,
                    "auc_roc": auc,
                    "auc_pr": pr,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": pred_seconds,
                    "paper_shape_match": matches,
                    "source": spec.source,
                    "note": spec.note,
                }
            )

        summary = {
            "dataset": dataset,
            "source": spec.source,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "anomaly_rate_pct": observed_rate,
            "threshold": threshold,
            "branch": args.branch,
            "runs": args.runs,
            "mean_auc_roc": float(np.mean(aucs)),
            "std_auc_roc": float(np.std(aucs)),
            "mean_auc_pr": float(np.mean(prs)),
            "std_auc_pr": float(np.std(prs)),
            "mean_fit_seconds": float(np.mean(fit_times)),
            "mean_predict_seconds": float(np.mean(pred_times)),
            "paper_shape_match": matches,
            "paper_n_samples": spec.paper.n_samples,
            "paper_n_features": spec.paper.n_features,
            "paper_anomaly_rate_pct": spec.paper.anomaly_rate_pct,
            "note": spec.note,
        }
        summary_rows.append(summary)

        print(
            f"  -> auc={summary['mean_auc_roc']:.4f}±{summary['std_auc_roc']:.4f} "
            f"pr={summary['mean_auc_pr']:.4f}±{summary['std_auc_pr']:.4f} "
            f"fit={summary['mean_fit_seconds']:.2f}s pred={summary['mean_predict_seconds']:.2f}s"
        )

    write_csv(Path(args.details_output), detail_rows)
    write_csv(Path(args.output), summary_rows)

    print(f"\nWrote summary: {args.output}")
    print(f"Wrote per-run details: {args.details_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
