#!/usr/bin/env python3

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
        "--run-workers",
        type=int,
        default=1,
        help=(
            "Runs to execute in parallel within each dataset (default: 1, sequential). "
            "Useful for large slow datasets. Cannot be combined with --workers > 1."
        ),
    )
    parser.add_argument(
        "--flat-workers",
        type=int,
        default=0,
        help=(
            "If > 0, run a single flat pool over all (dataset, run) combinations with "
            "this many workers. Maximizes utilization on a large CPU node by keeping "
            "every worker busy regardless of dataset boundaries. Overrides --workers "
            "and --run-workers."
        ),
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


_PATCHED_INDEX_MERGED_NODE = False


def _patch_optiforest_index_merged_node() -> None:
    """Vectorize ``HierTree.index_merged_node`` for ``num_of_bin == 2``.

    Upstream OptIForest's ``index_merged_node`` calls ``scipy.spatial.distance.euclidean``
    inside an O(n^2) Python double loop. For datasets where the threshold is
    in the hundreds, this makes each tree build take minutes — millions of
    individual scipy calls dominate the runtime (94% in cProfile).

    For ``num_of_bin == 2`` the upstream weighted-distance formula simplifies
    algebraically::

        dist_ij = ||C[i] - C[j]|| * 2 * S[i] * S[j] / (S[i] + S[j])

    which equals BOTH branches of the upstream code:
      * the ``S[i] == S[j]`` shortcut ``||C[i]-C[j]|| * S[i]`` matches because
        ``2*S*S/(2S) = S``;
      * the general ``get_weight_distance`` call expands to the same expression.

    The derivation uses
    ``newcenter = (S_i*C_i + S_j*C_j) / (S_i + S_j)``, giving
    ``newcenter - C_i = S_j/(S_i+S_j) * (C_j - C_i)`` for any norm, so
    ``S_i*||nc-C_i|| + S_j*||nc-C_j|| = ||C_i-C_j|| * 2*S_i*S_j/(S_i+S_j)``.

    The vectorized version computes the full weighted-distance matrix with a
    single ``pdist`` call (C-level scipy) instead of n*(n-1)/2 Python calls,
    yielding the same argmin pair up to floating-point rounding.

    For ``num_of_bin == 3`` the expansion uses
    ``<a-c, b-c> = (||a-c||^2 + ||b-c||^2 - ||a-b||^2) / 2`` to write each
    ``||nc - C_k||^2`` as a linear combination of pairwise squared distances
    only::

        T = S_i + S_j + S_h
        ||nc - C_i||^2 = (S_j(S_j+S_h) D[i,j] + S_h(S_j+S_h) D[i,h]
                          - S_j S_h D[j,h]) / T^2

    Then the loop over ``i`` (n iterations) does vectorized arithmetic over
    all ``(j, h)`` pairs with ``i < j < h``, replacing the upstream O(n^3)
    Python triple loop. The only scipy call is the single ``pdist`` for the
    squared-distance matrix.

    Both vectorizations assume Euclidean distance (the L2OPT variant). For
    cosine/cityblock the patch falls back to the original implementation.
    """
    global _PATCHED_INDEX_MERGED_NODE
    if _PATCHED_INDEX_MERGED_NODE:
        return
    from detectors import opt_tree  # pylint: disable=import-error
    from scipy.spatial import distance as _scipy_distance
    from scipy.spatial.distance import pdist, squareform

    original_index_merged_node = opt_tree.HierTree.index_merged_node

    def patched(self, hirenodes, num_of_bin):
        # Only vectorize when the tree uses Euclidean distance (the L2OPT
        # variant). Other variants (cosine, cityblock) fall back to the
        # original Python loop — they would need their own derivations.
        if getattr(self, "distance", None) is not _scipy_distance.euclidean:
            return original_index_merged_node(self, hirenodes, num_of_bin)

        n = len(hirenodes)
        if num_of_bin == 2:
            if n < 2:
                return None
            centers = np.asarray([h.get_center() for h in hirenodes], dtype=float)
            sizes = np.asarray([h.get_data_size() for h in hirenodes], dtype=float)
            D = squareform(pdist(centers))
            S_prod = sizes[:, None] * sizes[None, :]
            S_sum = sizes[:, None] + sizes[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                W = D * (2.0 * S_prod / S_sum)
            W[np.tril_indices(n)] = np.inf
            i, j = np.unravel_index(int(np.argmin(W)), W.shape)
            return [int(j), int(i)]

        if num_of_bin == 3:
            if n < 3:
                return original_index_merged_node(self, hirenodes, num_of_bin)
            centers = np.asarray([h.get_center() for h in hirenodes], dtype=float)
            sizes = np.asarray([h.get_data_size() for h in hirenodes], dtype=float)
            # Squared pairwise distances: D2[a,b] = ||C_a - C_b||^2
            D2 = squareform(pdist(centers, "sqeuclidean"))

            # For triplet (i, j, h) with i < j < h, the upstream formula is
            #   nc = (S_i C_i + S_j C_j + S_h C_h) / T,  T = S_i + S_j + S_h
            #   dist = ||nc - C_i|| S_i + ||nc - C_j|| S_j + ||nc - C_h|| S_h
            # Expanding ||nc - C_k||^2 in terms of pairwise squared distances
            # (using <a-c, b-c> = (||a-c||^2 + ||b-c||^2 - ||a-b||^2) / 2):
            #   ||nc - C_i||^2 = (S_j (S_j+S_h) D[i,j] + S_h (S_j+S_h) D[i,h]
            #                    - S_j S_h D[j,h]) / T^2
            #   ||nc - C_j||^2 = (S_i (S_i+S_h) D[i,j] + S_h (S_i+S_h) D[j,h]
            #                    - S_i S_h D[i,h]) / T^2
            #   ||nc - C_h||^2 = (S_i (S_i+S_j) D[i,h] + S_j (S_i+S_j) D[j,h]
            #                    - S_i S_j D[i,j]) / T^2
            best_dist = np.inf
            best_triplet = None
            # Loop over the smallest index i, vectorize over (j, h) with i<j<h.
            for i in range(n - 2):
                # Build (j, h) pairs with i < j < h
                jh = np.triu_indices(n - i - 1, k=1)
                jj = jh[0] + i + 1
                hh = jh[1] + i + 1
                S_i = sizes[i]
                S_j = sizes[jj]
                S_h = sizes[hh]
                T = S_i + S_j + S_h
                D_ij = D2[i, jj]
                D_ih = D2[i, hh]
                D_jh = D2[jj, hh]

                SP_jh = S_j + S_h
                SP_ih = S_i + S_h
                SP_ij = S_i + S_j

                inv_T2 = 1.0 / (T * T)
                n_i_sq = (S_j * SP_jh * D_ij + S_h * SP_jh * D_ih - S_j * S_h * D_jh) * inv_T2
                n_j_sq = (S_i * SP_ih * D_ij + S_h * SP_ih * D_jh - S_i * S_h * D_ih) * inv_T2
                n_h_sq = (S_i * SP_ij * D_ih + S_j * SP_ij * D_jh - S_i * S_j * D_ij) * inv_T2

                # Clamp tiny negatives from floating-point noise
                np.maximum(n_i_sq, 0.0, out=n_i_sq)
                np.maximum(n_j_sq, 0.0, out=n_j_sq)
                np.maximum(n_h_sq, 0.0, out=n_h_sq)

                dist = (
                    np.sqrt(n_i_sq) * S_i
                    + np.sqrt(n_j_sq) * S_j
                    + np.sqrt(n_h_sq) * S_h
                )
                if dist.size == 0:
                    continue
                local_idx = int(np.argmin(dist))
                if dist[local_idx] < best_dist:
                    best_dist = float(dist[local_idx])
                    best_triplet = [int(hh[local_idx]), int(jj[local_idx]), int(i)]

            return best_triplet

        return original_index_merged_node(self, hirenodes, num_of_bin)

    opt_tree.HierTree.index_merged_node = patched
    _PATCHED_INDEX_MERGED_NODE = True


def import_optiforest():
    ensure_upstream_repo()
    sys.path.insert(0, str(UPSTREAM_DIR))
    from detectors import OptIForest  # pylint: disable=import-error

    _patch_optiforest_index_merged_node()
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


def _run_single_worker(
    dataset: str,
    run_number: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Top-level worker for intra-dataset run parallelism.

    Loads data independently inside the worker process to avoid pickling large
    numpy arrays across process boundaries.
    """
    ensure_upstream_repo()
    sys.path.insert(0, str(UPSTREAM_DIR))
    from detectors import OptIForest as _OptIForest  # pylint: disable=import-error

    _patch_optiforest_index_merged_node()

    spec = PAPER_DATASETS[dataset]
    X, y = load_dataset(dataset, spec, force=False)
    threshold = select_threshold(dataset, X, args)
    matches = shape_matches_paper(X, y, spec)
    observed_rate = round(100.0 * float(y.mean()), 2)
    row = run_dataset_run(
        dataset, run_number, args, _OptIForest, X, y, spec, threshold, matches, observed_rate
    )
    write_csv_atomic(partial_run_path(args, dataset, run_number), [row])
    return row


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

    run_workers = getattr(args, "run_workers", 1) or 1

    if run_workers > 1 and len(pending_runs) > 1:
        print(
            f"[{dataset}] running {len(pending_runs)} pending run(s) "
            f"with {run_workers} worker(s) in parallel",
            flush=True,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=run_workers) as executor:
            future_to_run = {
                executor.submit(_run_single_worker, dataset, run_number, args): run_number
                for run_number in pending_runs
            }
            for future in concurrent.futures.as_completed(future_to_run):
                run_number = future_to_run[future]
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(f"[{dataset}] run {run_number} failed") from exc
                print(
                    f"[{dataset}] run {run_number}/{args.runs} done "
                    f"auc={row['auc_roc']:.4f} pr={row['auc_pr']:.4f} "
                    f"fit={row['fit_seconds']:.2f}s pred={row['predict_seconds']:.2f}s",
                    flush=True,
                )
    else:
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


def run_flat_pool(args: argparse.Namespace, pending: list[str], flat_workers: int) -> None:
    """Run all pending (dataset, run) combinations in a single flat process pool.

    Unlike the per-dataset pool, this keeps every worker busy across dataset
    boundaries: a fast dataset finishing early frees its slot for a remaining
    run of a slow dataset, instead of leaving the slot idle.
    """
    # Order datasets by paper sample count ascending so smaller datasets
    # finish quickly and free workers for the heavy ones, instead of the
    # heavy ones (e.g. cover) hogging every slot from the start.
    ordered_pending = sorted(pending, key=lambda d: PAPER_DATASETS[d].paper.n_samples)
    work: list[tuple[str, int]] = []
    for dataset in ordered_pending:
        migrate_legacy_partials(args, dataset)
        for run_number in missing_runs(args, dataset):
            work.append((dataset, run_number))

    total = len(work)
    print(
        f"Flat pool: {total} pending (dataset, run) job(s) across {flat_workers} worker(s)",
        flush=True,
    )
    if not work:
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=flat_workers) as executor:
        future_to_item = {
            executor.submit(_run_single_worker, dataset, run_number, args): (dataset, run_number)
            for dataset, run_number in work
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_item):
            dataset, run_number = future_to_item[future]
            try:
                row = future.result()
            except Exception as exc:
                raise RuntimeError(f"[{dataset}] run {run_number} failed") from exc
            done += 1
            print(
                f"[{dataset}] run {run_number}/{args.runs} done ({done}/{total}) "
                f"auc={row['auc_roc']:.4f} pr={row['auc_pr']:.4f} "
                f"fit={row['fit_seconds']:.2f}s pred={row['predict_seconds']:.2f}s",
                flush=True,
            )

    for dataset in pending:
        run_rows = completed_run_rows(args, dataset)
        if run_rows:
            summary = summarize_run_rows(args, dataset, PAPER_DATASETS[dataset], run_rows)
            write_csv_atomic(partial_summary_path(args, dataset), [summary])


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
    if args.flat_workers < 0:
        raise ValueError("--flat-workers must be >= 0")
    if args.flat_workers > 0:
        if args.workers is not None and args.workers > 1:
            print("Note: --flat-workers overrides --workers.")
        if args.run_workers > 1:
            print("Note: --flat-workers overrides --run-workers.")
    elif workers > 1 and args.run_workers > 1:
        print(
            "Warning: --workers > 1 and --run-workers > 1 would create nested process pools. "
            "Resetting --run-workers to 1."
        )
        args.run_workers = 1

    print(
        f"Running OptIForest {args.version} on {len(selected)} dataset(s): "
        f"{', '.join(selected)}"
    )
    print(f"Using {workers} dataset worker(s); partials: {Path(args.partials_dir) / safe_version(args.version)}")

    ensure_upstream_repo()

    for dataset in selected:
        ensure_dataset_file(dataset, PAPER_DATASETS[dataset], force=args.force_refresh_data)

    completed = [dataset for dataset in selected if partial_is_complete(args, dataset)]
    pending = [dataset for dataset in selected if dataset not in completed]

    for dataset in completed:
        print(f"[{dataset}] skipping existing partial result", flush=True)

    if pending and args.flat_workers > 0:
        run_flat_pool(args, pending, args.flat_workers)
    elif pending and workers == 1:
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
