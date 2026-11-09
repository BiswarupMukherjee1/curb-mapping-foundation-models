# Results by experiment

Each section gives the question, the setup, where the numbers are, and what was learned, including what did not work.
All scores are on the test blocks at the 10 cm tolerance unless stated. Pessac = tile `OH`, Bordeaux centre = tile `RL`.

## 01 Data audit (`experiments/01_data_audit/`)

**Question.** Are the open orthophoto and PCRS accurate enough to measure 10 cm, and how much curb is visible from the
air?

**Setup.**
- `alignment.py`: RGB ↔ infrared co-registration by phase correlation. Orthophoto ↔ PCRS offset as the shift that
  maximises image gradient energy under the curb lines (grid search, parabolic sub-pixel peak), globally and per
  delivery zone.
- `visibility.py`: per curb point, NDVI from the infrared orthophoto and height above ground from the surface models.
  Points with canopy (NDVI > 0.2) or a raised object (> 0.8 m) are hidden; dark points are visible but in shadow. The
  offset is re-estimated on visible curbs only, and validated by recovering an injected shift.

**Results** (`results/data_audit.json`).
- Offset on visible curbs: 0.2 / 1.8 cm (east / north) in Pessac, 0.9 / 2.5 cm in Bordeaux centre. An injected 15 cm /
  −10 cm shift is recovered to within 0.6 cm.
- Visible from the air: 79 % (Pessac), 73 % (Bordeaux centre). With a stricter neighbourhood rule (no raised object
  within 1.5 m): 63 % and 50 %.
- The RGB world files are offset by half a pixel from the tile grid; the loader snaps them.
- Three delivery zones in Bordeaux centre are consistently 43–51 cm off the imagery (C12, which the city lists as
  being re-acquired, and BC02, BC03, listed as complete); two more (30, BC06) give unstable offsets (their two
  half-samples disagree by 25–35 cm). These five are excluded from the evaluation area and used as real cases in
  experiment 05.

## 02 Classical baseline (`experiments/02_classical_baseline/`)

**Question.** How far does line detection without learning get?

**Setup.** LSD line segments on luminance; per segment: length, contrast across the line, NDVI on the line, height at
the midpoint, optionally a parallel partner 10–40 cm away. A grid of 48 threshold combinations is searched on the
validation blocks.

**Results** (`results/classical_baseline.json`). Completeness / correctness / quality at 10 cm: Pessac 0.29 / 0.20 /
0.15, Bordeaux centre 0.14 / 0.12 / 0.07. Curbs under trees are almost never found (3 %).

**Note.** The first protocol counted every line drawn outside the PCRS coverage (gardens, courtyards) as false. Those
areas are not surveyed, so the evaluation area was restricted to the PCRS coverage; all experiments use the restricted
protocol (`src/pcrs_ai/protocol.py`).

## 03 Learned models (`experiments/03_learned_models/`)

**Question.** Do frozen foundation-model features help, compared with the same network on colour alone and with a
fine-tuned U-Net?

**Setup.** `train.py` (see README section 4); `rescore.py` re-selects every model's threshold on one common
validation grid; `scripts/compare_runs.py` computes the block-bootstrap intervals.

**Results** (`results/models.csv`, `results/main_comparison.csv`).

Equal training (all labels, 6000 iterations), quality at 10 cm with 95 % block-bootstrap intervals:

| model | Pessac | Bordeaux centre | second seed |
|---|---|---|---|
| U-Net ResNet-34, fine-tuned | 0.55 [0.48, 0.62] | 0.30 [0.26, 0.35] | 0.545 / 0.298 |
| DINOv3 ViT-L, satellite | 0.54 [0.47, 0.62] | 0.30 [0.25, 0.35] | 0.539 / 0.297 |
| DINOv3 ViT-B, web | 0.52 [0.46, 0.59] | 0.29 [0.25, 0.33] | |
| DOFA ViT-B | 0.41 [0.35, 0.47] | 0.17 [0.13, 0.22] | |
| AnySat | 0.37 [0.31, 0.44] | 0.17 [0.12, 0.22] | |
| colour-only network | 0.31 [0.26, 0.35] | 0.13 [0.10, 0.16] | |

- The paired difference U-Net − DINOv3 is +0.010 [−0.007, +0.026] in Pessac and −0.002 [−0.023, +0.017] in Bordeaux
  centre: no measurable difference.
- DINOv3 − colour-only: +0.235 [+0.202, +0.273] and +0.173 [+0.141, +0.204].
- DINOv3 web − satellite: −0.018 [−0.037, +0.001] and −0.014 [−0.033, +0.004].
- DOFA and AnySat are 0.13–0.17 below DINOv3 and 0.04–0.10 above colour-only (all intervals exclude zero).
- On curbs visible from the air, completeness at 10 cm is 0.75 (Pessac) and 0.54 (Bordeaux centre) for both DINOv3 and
  the U-Net.

**Design choices that mattered.**
- *Two label channels.* With one band for both curb edges, the network drew a single line between them, about 7 cm from
  each. Separate bottom and top channels raised Pessac completeness from 0.40 to 0.57 (DINOv3).
- *Equal training length.* After 2000 iterations the frozen-encoder models were not converged (DINOv3 0.43, DOFA 0.16
  in Pessac) and a U-Net trained for 6000 iterations looked clearly better. All models were therefore trained for 6000
  iterations, and only those runs are compared.

## 04 Review ranking and error analysis (`experiments/04_review_ranking/`)

**Question.** Can each drawn line carry a confidence and a reason code so that a drafter checks the risky lines first?
What are the errors?

**Setup.** Model: fine-tuned U-Net. Per drawn line: share of its length within 10 cm of a
PCRS line of the same class, probability statistics along it, length, visibility features, and a reason code. Two
confidences: the mean probability, and a logistic model of "correct" fitted on validation lines. `error_analysis.py`
classifies every drawn point by what lies under it, and tests two post-processing fixes.

**Results** (`results/review_ranking.json`, `results/error_analysis.json`).
- Wrong drawn length: 35 % (Pessac), 58 % (Bordeaux centre).
- Reviewing the least confident 10 / 20 / 30 % of the length catches 21 / 35 / 48 % of the wrong length in Pessac and
  15 / 29 / 42 % in Bordeaux (calibrated confidence; random order 10 / 20 / 30 %). The pre-set target (50 % at 20 %)
  is not met; the calibrated model adds nothing over the mean probability.
- Reason codes: see README section 6.4.
- Error decomposition and the two rejected fixes: README section 6.3.

## 05 Map quality control (`experiments/05_map_quality_control/`)

**Question.** Can the model check an existing PCRS against new imagery?

**Setup.** Support of a map line = mean probability of its class along it (max over ±1 px). (A) Test-area curbs cut
into 5 m pieces, a random half shifted perpendicular by 5 / 10 / 20 / 50 cm. (B) 10 % of the pieces deleted; model
lines with no map line within 20 cm are candidates. (C) Support per delivery zone outside the training blocks, compared
with the city's zone status and with the zones flagged by the audit.

**Results** (`results/map_quality_control.json`).

| shift | AUROC, visible pieces | AUROC, all pieces |
|---|---|---|
| 5 cm | 0.54 | 0.53 |
| 10 cm | 0.61 | 0.58 |
| 20 cm | 0.79 | 0.73 |
| 50 cm | 0.96 | 0.91 |

- At the support threshold that catches 80 % of the 20 cm shifts, 32 % of the untouched visible pieces are flagged.
- Deleted lines: 47 % / 39 % of the deleted length recovered, 174 / 304 m of false candidates per km of map.
- Zone support: complete 0.60, being re-acquired 0.48, needs re-acquisition 0.08. The zones flagged by the audit do not
  rank consistently lowest (AUROC 0.60): support also falls where curbs are simply hard to see, so a zone score would
  need normalising by visibility.

## 06 Explainability (`experiments/06_explainability/`)

**Question.** What does each trained model rely on?

**Setup** (`run.py`). (A) Branch test, frozen-encoder models only: on the test blocks the trained head is run with the
encoder features set to zero, or with its pixel input replaced by a flat grey image, and scored at 10 cm; the head is not
retrained. (B) RISE (Petsiuk et al., 2018) for 15 correct and 15 wrong drawn lines per area and model: the score is the
mean probability of the line's class along the line; 500 random masks on an 8 × 8 grid (3.2 m cells), masked pixels
set to the window's mean colour. Per line: share of positive attribution within 0.5 m and within 0.5–3 m of the line,
and the ratio of their densities per unit area.

**Results** (`results/explainability.json`; models trained for 6000 iterations).
- Branch test, DINOv3-sat: features only 0.49 / 0.27 (normal 0.54 / 0.30); pixel branch only 0.00 / 0.00.
- RISE density ratio, median: colour-only 1.76, DINOv3 1.68, U-Net 1.59; above 1 for 88–95 % of lines. Share of positive
  attribution within 0.5 m: 6–8 %; within 0.5–3 m: 36–43 %.
- Correct and wrong lines do not differ (Mann–Whitney p = 0.23–0.30 for the density ratio).

**Interpretation.** The frozen-encoder model relies on the DINOv3 features almost entirely, and those features carry
the curb position to well under their 80 cm patch size. All models use the street context within a few metres. RISE at
3.2 m cells cannot explain errors of a few centimetres. Grad-CAM-type methods were not used: they summarise one whole
feature layer at the 80 cm patch grid or coarser, which is further from the error scale than RISE.
