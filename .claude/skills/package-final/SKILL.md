---
name: package-final
description: Build the final competition zip (<team>_submission.zip) — output TSVs from the chosen submission, a self-contained runnable code copy with README and pinned requirements, and the filled Documentation_template.md; verify structure and reproducibility. Human-invoked, Day 3.
argument-hint: "<sub_id of the final submission> <team_name>"
disable-model-invocation: true
---

# Package the final submission: $ARGUMENTS

Target structure (exact):
```
<team_name>_submission.zip
├── output/matching_results.tsv          (identical bytes to the uploaded final file)
├── output/candidate_pairs.tsv           (exact set the final model scored)
├── code/business_entity_resolution/src/ (all source: src/ber, pipelines/, scripts/ needed)
├── code/business_entity_resolution/README.md
├── code/business_entity_resolution/requirements.txt  (pinned ==)
└── Documentation_template.md            (filled)
```

Steps:
1. Build it in `dist/<team_name>_submission/` (git-ignored). Copy the output TSVs from `submissions/files/<sub_id>/`. Run the official validator on them again, with `--candidate`.
2. Code copy:
   - `src/ber`, the final `pipelines/*.py`, and the `scripts/*.py` used → `code/business_entity_resolution/src/`.
   - Keep the relative imports working (`_bootstrap.py` path) and update `ber/paths.py` defaults if needed (env vars `AMLC_RAW_DIR` etc.).
   - Remove dead experiments.
   - No data, no artifacts, no secrets, no absolute paths.
3. `requirements.txt`: `python -m pip freeze` filtered to the packages actually imported (grep the imports). Pin `==`. Record the Python version and the GPU/CUDA used in the README.
4. README.md: exact end-to-end commands from raw data to both TSVs:
   1. prepare
   2. normalize
   3. train (with seeds)
   4. blocking
   5. scoring
   6. decision
   7. write outputs

   Include the hardware used and the runtime per step. List pretrained models with their **licenses (MIT/Apache) and parameter counts**, and all hand-written dictionaries.
5. Documentation_template.md: start from `student_resource/Documentation_template.md` (copy; don't edit the original). Fill every section from `docs/strategy.md`, `docs/decisions.md`, the final run records and the error analyses:
   - blocking keys + candidate count + recall
   - features
   - model
   - threshold method
   - val F0.5 by country
   - common FPs/FNs

   About 2 pages of main text plus an appendix. State clearly that no external data or lookups were used.
6. Reproducibility smoke test: in a fresh temporary folder, run the README commands on the `micro` subset (or a dry run) to catch missing files/imports.
7. Zip it as `<team_name>_submission.zip`. Verify with `python -m zipfile -l`. Report the size and the tree to the human.
