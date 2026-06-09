#!/usr/bin/env python3
"""Runner do estudo de reprodução do OptIForest (IJCAI 2023).

Para cada dataset selecionado, baixa/prepara os dados (ADBench, OpenML ou o
``ad.csv`` do repo upstream), executa o OptIForest ``--runs`` vezes com a
configuração do artigo (100 árvores, branch=0, limiar ε por tamanho) e escreve
métricas AUC-ROC/AUC-PR por execução e agregadas.

Os resultados são gravados de forma incremental em ``partial/<versao>/<dataset>/``
(um CSV por run + summary), permitindo retomar execuções interrompidas sem
recomputar runs já concluídos. Os CSVs finais consolidados só são escritos após
todos os datasets terminarem. Veja ``--help`` para os argumentos de CLI.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
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
        (
            "annthyroid",
            DatasetSpec(
                kind="npz",
                cache_name="2_annthyroid.npz",
                remote_name="2_annthyroid.npz",
                paper=PaperStats(7200, 6, 7.42),
                source="adbench",
                note=(
                    "New dataset NOT in the OptIForest paper. Medical (thyroid disease) "
                    "benchmark from ADBench; low-dimensional regime distinct from arrhythmia. "
                    "paper_shape_match compares against ADBench's own stats, not the paper."
                ),
            ),
        ),
        (
            "fraud",
            DatasetSpec(
                kind="npz",
                cache_name="13_fraud.npz",
                remote_name="13_fraud.npz",
                paper=PaperStats(284807, 29, 0.17),
                source="adbench",
                note=(
                    "New dataset NOT in the OptIForest paper. Credit-card fraud (finance) "
                    "benchmark from ADBench; ultra-rare anomaly rate (0.17%) extrapolates "
                    "beyond the paper's range. Stats are ADBench's own, not the paper."
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
        "--partials-dir",
        default=str(RESULTS_DIR / "partial"),
        help="Directory for per-run checkpoint CSVs.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Experiment/code version namespace for partial results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Datasets to run in parallel. Defaults to min(6, CPU count, dataset count).",
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


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_csv(tmp_path, rows)
    tmp_path.replace(path)


def safe_version(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe or "v1"


def partial_dataset_dir(args: argparse.Namespace, dataset: str) -> Path:
    return Path(args.partials_dir) / safe_version(args.version) / dataset


def partial_run_path(args: argparse.Namespace, dataset: str, run_number: int) -> Path:
    return partial_dataset_dir(args, dataset) / f"run_{run_number:02d}.csv"


def partial_summary_path(args: argparse.Namespace, dataset: str) -> Path:
    return partial_dataset_dir(args, dataset) / "summary.csv"


def run_row_is_complete(
    args: argparse.Namespace, dataset: str, run_number: int, row: dict[str, str]
) -> bool:
    threshold_arg = "" if args.threshold is None else str(args.threshold)
    try:
        return (
            row.get("dataset") == dataset
            and row.get("version") == args.version
            and int(row.get("run", -1)) == run_number
            and int(row.get("runs", -1)) == args.runs
            and int(row.get("trees", -1)) == args.trees
            and int(row.get("branch", -1)) == args.branch
            and int(row.get("seed_base", -1)) == args.seed_base
            and row.get("threshold_mode") == args.threshold_mode
            and row.get("threshold_arg", "") == threshold_arg
        )
    except ValueError:
        return False


def completed_run_rows(args: argparse.Namespace, dataset: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_number in range(1, args.runs + 1):
        run_rows = csv_rows(partial_run_path(args, dataset, run_number))
        if len(run_rows) == 1 and run_row_is_complete(
            args, dataset, run_number, run_rows[0]
        ):
            rows.append(run_rows[0])
    return rows


def missing_runs(args: argparse.Namespace, dataset: str) -> list[int]:
    complete = {
        int(row["run"])
        for row in completed_run_rows(args, dataset)
    }
    return [run_number for run_number in range(1, args.runs + 1) if run_number not in complete]


def partial_is_complete(args: argparse.Namespace, dataset: str) -> bool:
    return not missing_runs(args, dataset)


def legacy_partial_paths(args: argparse.Namespace, dataset: str) -> tuple[Path, Path]:
    version_dir = Path(args.partials_dir) / safe_version(args.version)
    return (
        version_dir / f"{dataset}_summary.csv",
        version_dir / f"{dataset}_runs.csv",
    )


def migrate_legacy_partials(args: argparse.Namespace, dataset: str) -> None:
    summary_path, details_path = legacy_partial_paths(args, dataset)
    if not summary_path.exists() or not details_path.exists():
        return
    detail_rows = csv_rows(details_path)
    if len(detail_rows) != args.runs:
        return
    for row in detail_rows:
        try:
            run_number = int(row["run"])
        except (KeyError, ValueError):
            return
        if not run_row_is_complete(args, dataset, run_number, row):
            return
    for row in detail_rows:
        write_csv_atomic(partial_run_path(args, dataset, int(row["run"])), [row])
    summary_rows = csv_rows(summary_path)
    if len(summary_rows) == 1:
        write_csv_atomic(partial_summary_path(args, dataset), summary_rows)


def run_dataset_run(
    dataset: str,
    run_number: int,
    args: argparse.Namespace,
    OptIForest,
    X: np.ndarray,
    y: np.ndarray,
    spec: DatasetSpec,
    threshold: int,
    matches: bool,
    observed_rate: float,
) -> dict[str, object]:
    seed = args.seed_base + run_number - 1
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

    return {
        "version": args.version,
        "dataset": dataset,
        "run": run_number,
        "seed": seed,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "anomaly_rate_pct": observed_rate,
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "threshold_arg": "" if args.threshold is None else args.threshold,
        "branch": args.branch,
        "trees": args.trees,
        "runs": args.runs,
        "seed_base": args.seed_base,
        "auc_roc": auc,
        "auc_pr": pr,
        "fit_seconds": fit_seconds,
        "predict_seconds": pred_seconds,
        "paper_shape_match": matches,
        "source": spec.source,
        "note": spec.note,
    }


def summarize_run_rows(
    args: argparse.Namespace,
    dataset: str,
    spec: DatasetSpec,
    run_rows: list[dict[str, object] | dict[str, str]],
) -> dict[str, object]:
    first = run_rows[0]
    return {
        "version": args.version,
        "dataset": dataset,
        "source": spec.source,
        "n_samples": int(first["n_samples"]),
        "n_features": int(first["n_features"]),
        "anomaly_rate_pct": float(first["anomaly_rate_pct"]),
        "threshold": int(first["threshold"]),
        "threshold_mode": args.threshold_mode,
        "threshold_arg": "" if args.threshold is None else args.threshold,
        "branch": args.branch,
        "trees": args.trees,
        "runs": args.runs,
        "seed_base": args.seed_base,
        "completed_runs": len(run_rows),
        "mean_auc_roc": float(np.mean([float(row["auc_roc"]) for row in run_rows])),
        "std_auc_roc": float(np.std([float(row["auc_roc"]) for row in run_rows])),
        "mean_auc_pr": float(np.mean([float(row["auc_pr"]) for row in run_rows])),
        "std_auc_pr": float(np.std([float(row["auc_pr"]) for row in run_rows])),
        "mean_fit_seconds": float(np.mean([float(row["fit_seconds"]) for row in run_rows])),
        "mean_predict_seconds": float(
            np.mean([float(row["predict_seconds"]) for row in run_rows])
        ),
        "paper_shape_match": first["paper_shape_match"],
        "paper_n_samples": spec.paper.n_samples,
        "paper_n_features": spec.paper.n_features,
        "paper_anomaly_rate_pct": spec.paper.anomaly_rate_pct,
        "note": spec.note,
    }


def run_dataset(dataset: str, args: argparse.Namespace) -> dict[str, object]:
    migrate_legacy_partials(args, dataset)

    OptIForest = import_optiforest()

    spec = PAPER_DATASETS[dataset]
    X, y = load_dataset(dataset, spec, force=False)
    threshold = select_threshold(dataset, X, args)
    matches = shape_matches_paper(X, y, spec)
    observed_rate = round(100.0 * float(y.mean()), 2)

    print(
        f"[{dataset}] shape={X.shape} anomalies={int(y.sum())}/{len(y)} "
        f"rate={observed_rate:.2f}% threshold={threshold} match_paper={matches}",
        flush=True,
    )

    pending_runs = missing_runs(args, dataset)
    completed_count = args.runs - len(pending_runs)
    if completed_count:
        print(f"[{dataset}] found {completed_count}/{args.runs} completed run(s)", flush=True)

    for run_number in pending_runs:
        print(f"[{dataset}] run {run_number}/{args.runs} starting", flush=True)
        row = run_dataset_run(
            dataset,
            run_number,
            args,
            OptIForest,
            X,
            y,
            spec,
            threshold,
            matches,
            observed_rate,
        )
        write_csv_atomic(partial_run_path(args, dataset, run_number), [row])
        print(
            f"[{dataset}] run {run_number}/{args.runs} "
            f"auc={row['auc_roc']:.4f} pr={row['auc_pr']:.4f} "
            f"fit={row['fit_seconds']:.2f}s pred={row['predict_seconds']:.2f}s",
            flush=True,
        )

    run_rows = completed_run_rows(args, dataset)
    summary = summarize_run_rows(args, dataset, spec, run_rows)
    write_csv_atomic(partial_summary_path(args, dataset), [summary])

    print(
        f"  -> [{dataset}] auc={summary['mean_auc_roc']:.4f}±{summary['std_auc_roc']:.4f} "
        f"pr={summary['mean_auc_pr']:.4f}±{summary['std_auc_pr']:.4f} "
        f"fit={summary['mean_fit_seconds']:.2f}s pred={summary['mean_predict_seconds']:.2f}s",
        flush=True,
    )
    return summary


def collect_partials(
    args: argparse.Namespace, selected: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    for dataset in selected:
        run_rows = completed_run_rows(args, dataset)
        summaries = csv_rows(partial_summary_path(args, dataset))
        if not summaries and len(run_rows) == args.runs:
            summary = summarize_run_rows(args, dataset, PAPER_DATASETS[dataset], run_rows)
            write_csv_atomic(partial_summary_path(args, dataset), [summary])
            summaries = csv_rows(partial_summary_path(args, dataset))
        summary_rows.extend(summaries)
        detail_rows.extend(run_rows)
    return summary_rows, detail_rows


def default_workers(dataset_count: int) -> int:
    return max(1, min(6, os.cpu_count() or 1, dataset_count))


def main() -> int:
    args = parse_args()
    selected = dataset_list(args.datasets)
    workers = args.workers if args.workers is not None else default_workers(len(selected))
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.trees < 1:
        raise ValueError("--trees must be at least 1")
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    print(
        f"Running OptIForest {args.version} on {len(selected)} dataset(s): "
        f"{', '.join(selected)}"
    )
    print(f"Using {workers} dataset worker(s); partials: {Path(args.partials_dir) / safe_version(args.version)}")

    for dataset in selected:
        ensure_dataset_file(dataset, PAPER_DATASETS[dataset], force=args.force_refresh_data)

    completed = [dataset for dataset in selected if partial_is_complete(args, dataset)]
    pending = [dataset for dataset in selected if dataset not in completed]

    for dataset in completed:
        print(f"[{dataset}] skipping existing partial result", flush=True)

    if pending and workers == 1:
        for dataset in pending:
            run_dataset(dataset, args)
    elif pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_dataset = {
                executor.submit(run_dataset, dataset, args): dataset for dataset in pending
            }
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"{dataset} failed") from exc

    summary_rows, detail_rows = collect_partials(args, selected)

    if summary_rows:
        write_csv(Path(args.output), summary_rows)
    if detail_rows:
        write_csv(Path(args.details_output), detail_rows)

    print(f"\nWrote summary: {args.output}")
    print(f"Wrote per-run details: {args.details_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
