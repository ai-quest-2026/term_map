# GPU runbook: staged v2 run with checkpoints and gates

Every step prints a one-line result as soon as it is known, writes JSON to `reports/stages/<run>/`, and **stops
on its own if a gate fails** (exit code 2). Fold 0 of the 5-fold split is the smoke test. Never start the
full 5-fold run before fold 0 has passed all gates. Keep each step's log: `2>&1 | tee logs/<name>.log`.

## 0. Setup (10 min)
```bash
cd ~/term_map && git pull && source .venv/bin/activate && mkdir -p logs
pkill -f "[v]llm serve" || true        # free the GPU; [v] stops pkill matching its own shell
pip install peft accelerate             # then confirm torch is unchanged:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m scripts.doctor                # must end with "0 FAIL"
# second terminal, only if the LLM selector (step 4) will be tried; ~15 GB:
hf download Qwen/Qwen2.5-7B-Instruct
```
Baseline for comparisons: the per-row output of the locked 5-fold run (#8: ICD 0.658 / SNOMED 0.604).
```bash
head -1 reports/val_predictions.csv     # must contain a "fold" column
cp reports/val_predictions.csv reports/baseline_cv5_locked.csv
```
If that file was overwritten by a later run, skip `--baseline-csv` below and compare against the numbers in
`docs/experiments.md`.

## 1. S1+S2 smoke on fold 0: retrieval fixes + regularised ranker (~20 min; first run also rebuilds dictionary caches)
```bash
export TERMMAP_PROGRESS=500
python -m scripts.pipeline --run v2 --folds 0 --stages s2 --baseline-csv reports/baseline_cv5_locked.csv 2>&1 | tee logs/v2_s2_f0.log
```
The first run rebuilds `codesystem_*_v2` (fixed SNOMED tags, new ICD wording). SapBERT re-encodes only the
changed texts. Watch for:
- `s2 fold 0 [ICD10CM] ... r@30=... r@any=... unseen r@any=...`
- `train top1 oof/in-sample=...`: the gap should be much smaller than before (0.57 / 0.86).
- **Gate:** candidate recall must be ≥ the locked run's fold-0 recall − 1 point, on both systems.

## 2. S3 smoke: cross-encoder v2 on fold 0 (~15–25 min)
```bash
python -m scripts.pipeline --run v2 --folds 0 --stages s3 2>&1 | tee logs/v2_s3_f0.log
```
- Prints dev accuracy after **every epoch**. **Gate:** epoch-1 dev accuracy must be ≥ 0.45, otherwise training aborts
  within minutes. For reference, CE v1 alone was 0.48 / 0.50.
- Then prints: `gold@30`, `ranker`, `ce`, `blend0.3`, `best_blend`. **Gate:** best blend ≥ ranker.
- Optional, comparing a different starting model on the *same* candidate lists:
  ```bash
  TERMMAP_CE_INIT=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    python -m scripts.pipeline --run v2_bmb --s2-from v2 --folds 0 --stages s3 2>&1 | tee logs/v2_bmb_s3_f0.log
  ```

## 3. S5 on fold 0: stacker + training-term lookup, then the paired comparison (2 min)
```bash
python -m scripts.pipeline --run v2 --folds 0 --stages s5 2>&1 | tee logs/v2_s5_f0.log
python -m scripts.compare reports/baseline_cv5_locked.csv reports/stages/v2/predictions.csv --slices
```
`compare` matches the fold-0 rows only and prints each system's accuracy for A (locked) and B (v2), the difference
with a 95% CI, and the McNemar p-value.

**Decision.** On ~1,000 terms per system, only ≥3-point differences are detectable.
- If B ≥ A on both systems (or within −1 point on one with a clear gain on the other): run all folds (step 5).
- If B is significantly worse: stop and send me the logs.

## 4. (Should) LLM selector on fold 0 (~1–2 h; person B, in parallel with step 5)
```bash
python -m scripts.train_selector --run v2 --fold 0 2>&1 | tee logs/sel_f0.log
```
- Every 25 steps it prints `loss` and the ETA.
- At step 150 it prints `CHECKPOINT ... selector dev acc X vs ranker Y`.
- **Gate:** it stops itself if X < Y − 0.05.
- At the end it prints `selector top-1` vs `ranker` on fold 0. Then run:
```bash
python -m scripts.pipeline --run v2 --folds 0 --stages s5 --with-selector 2>&1 | tee logs/v2_s5_sel_f0.log
```
- **Keep the selector only if** the `FINAL` line beats the step-3 FINAL by ≥ 2 points on the headline system.
- Running alongside step 5 is fine: the 7B model uses about 30 GB; the cross-encoder uses about 8 GB.

## 5. Full 5-fold CV of v2 (~1.5–2 h; fold 0 is cached)
```bash
python -m scripts.pipeline --run v2 --folds all --stages s2,s3,s5 --baseline-csv reports/baseline_cv5_locked.csv --export-stacker 2>&1 | tee logs/v2_cv5.log
python -m scripts.compare reports/baseline_cv5_locked.csv reports/stages/v2/predictions.csv --slices | tee logs/v2_vs_locked.txt
```
- The gates still run per fold.
- `--export-stacker` writes `data/cache/stacker_<system>.joblib`, which `predict.py` uses.
- **Ship v2 if** the pooled headline beats 0.604 and McNemar p < 0.05 (or ≥ +1.5 points).

## 6. Final predictions (~20 min)
```bash
python -m scripts.predict --input <eval.csv> --output predictions.csv --details details.csv --refit
python -m scripts.predict --input <eval.csv> --output predictions_b.csv --refit && cmp predictions.csv predictions_b.csv && echo DETERMINISTIC
```
Add `--selector data/cache/stages/v2/selector_f0` to both commands only if step 4 passed **and** the stacker was
exported with `--with-selector`.

## Times (A6000, estimates)
| Step | Time |
|---|---|
| Dictionary rebuild (once) | ~10 min |
| s2 per fold | ~8 min |
| s3 per fold (4 epochs, k=30, 2 systems) | ~10–15 min |
| s5 | 1 min |
| Selector | ~25 min/epoch on ~4k terms |

## What each gate protects against
| Gate | Catches |
|---|---|
| s2 recall | a retrieval change that silently drops gold codes |
| s3 epoch-1 dev | a cross-encoder that isn't learning (bad inputs, LR, init) — aborts after minutes instead of hours |
| s3 blend ≥ ranker | a cross-encoder that trains but adds nothing |
| selector step-150 check | an LLM fine-tune that underperforms the ranker — stops before the remaining ~1 h |
| compare | "improvements" that are noise |
