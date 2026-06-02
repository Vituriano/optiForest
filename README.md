# OptIForest Reproduction

This workspace reproduces the released OptIForest implementation against the paper's 20 benchmark datasets as closely as the public artifacts allow.

## Setup

The upstream code was released for `numpy==1.20.1`, `sklearn==0.22.1`, and `pandas==1.4.1`. The reproduction runner requires `numpy<2` — numpy 2.x removed `np.mat` which the upstream code uses.

**Tested configuration:** Python 3.12, pip, venv.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Or with uv (Python 3.11 also works):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

The runner bootstraps the upstream OptIForest repo if `upstream_optiforest/` is missing, downloads the benchmark datasets it can source directly, caches them under `data/benchmarks/`, and writes results to `results/`.

**Full paper run** (all 20 datasets, 15 runs each — takes several hours):

```bash
.venv/bin/python scripts/run_optiforest_study.py --datasets all --runs 15 2>&1 | tee results/full_run.log
```

**Smoke test** (uses cached datasets, completes in ~1 min):

```bash
.venv/bin/python scripts/run_optiforest_study.py --datasets cardio --runs 1
```

**Subset run:**

```bash
.venv/bin/python scripts/run_optiforest_study.py --datasets ad,arrhythmia,cardio --runs 2
```

Results are written to `results/optiforest_summary.csv` and `results/optiforest_runs.csv` **only after all datasets finish**. A mid-run crash will not overwrite existing CSVs.

## Notes

- The paper's Appendix C implies `100` trees, `branch=0`, and a threshold rule of:
  - `512` for `ad` and `vowel`
  - `403` for datasets with more than `10,000` samples or more than `1,000` features
  - `55` otherwise
- Several released benchmark files do not exactly match the appendix table. The runner records whether each dataset shape matches the paper and carries a note when it does not.
- The paper cites `KDDCup99`, but the released implementation and public benchmark bundles do not expose a matching single-file loader. The harness currently uses the paper's separate `http` dataset instead for a runnable fallback and records that explicitly in the output.
