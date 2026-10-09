# Model identity and experiment boundaries

- LGMRecOpt is a transparent extension of LGMRec, not an independent baseline.
- Only TRAIN interactions may construct graphs, sample negatives, or train parameters.
- Select models on validation; evaluate test only after the final configuration is frozen.
- Compare models with identical data fingerprints, seeds, and shared training settings.
- Preserve existing source, data, logs, and checkpoints. Never overwrite experiment directories.
- Distinguish implementation, ablation evidence, and stable performance improvements.
