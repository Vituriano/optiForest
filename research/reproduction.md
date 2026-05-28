# Reproduction Notes

This repo uses the public OptIForest codebase at [xiagll/OptIForest](https://github.com/xiagll/OptIForest) and reconstructs the study runner around it.

## Dataset mapping

The public OptIForest repo ships only `ad.csv`. The rest of the benchmark set is reconstructed from public benchmark repositories:

- Most tabular datasets come from the ADBench classical dataset bundle.
- `arrhythmia` is reconstructed from OpenML by dropping the 5 columns with missing values, recovering the paper's 274-feature shape.
- `ionosphere` is reconstructed from OpenML with the full 34-feature representation.

## Known gaps against the paper

- `waveform`: the public ADBench file is `3443 x 21` with `2.90%` anomalies; the paper appendix lists `3505 x 21` and `4.62%`.
- `wine`: the public ADBench file is the classic `129 x 13` ODDS dataset; the paper appendix lists `5318 x 11`.
- `KDDCup99`: the paper includes it as a separate row, but the released code and accessible public bundles do not provide a matching ready-to-run file in the same format as the other datasets.

The runner does not hide these discrepancies. It records the observed dataset shape and a `paper_shape_match` flag in the output CSVs.
