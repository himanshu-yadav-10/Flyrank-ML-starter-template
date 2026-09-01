# Capstone Report — Refresh / Content Opportunity Scoring (Lane 2)

- **Author:** FlyRank ML Intern (Himanshu Yadav)
- **Lane:** 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/himanshu-yadav-10/Flyrank-ML-starter-template
- **Deployed paper:** https://himanshu-yadav-10.github.io/Flyrank-ML-starter-template/
- **Date:** 2026-09-01

> This report mirrors the deployed research paper. The paper (docs/index.html) carries the
> canonical 9-section version; this file is the graded repo-side report. All numbers here
> reproduce from a fresh clone (see §8).

## 0. Abstract

A content team must decide which pages to refresh first. Using the FlyRank ML Internship
dataset — a pseudonymized 30,000-page slice of real search performance across 32 clients — I
trained on 26 clients and held out 6 unseen clients. I scored each page for observed decline
risk (last-30-day impressions falling) with a gradient-boosting classifier on leakage-safe
signals, and compared it to a transparent "stale + visible" rule on the same split. On held-out
clients the model reaches precision@50 of **0.86** vs **0.32** for the rule and **0.39** for random
ordering (~2.7× over the rule). The output is a ranked content-opportunity queue with reason
codes that tells an editor which pages to refresh, rewrite, review, or monitor.

## 1. Problem framing

The decision this supports is **ordering the review backlog** so finite editing hours hit the
highest-value pages first. Unit of analysis: one content page. Output: a probability of decline
plus a human-facing reason code and an action. The cost of a wrong call: a genuinely declining
high-traffic page buried in the queue loses traffic while it waits, or scarce effort is spent
refreshing the wrong page. Data/ML help because no single observed signal separates decliners
from stable pages (each lone signal's AUC ≈ 0.4–0.5); a model reading them together orders them
far better than a hand-written rule.

## 2. Data safety

Used the anonymized starter slice: **30,000 pages × 44 raw columns**, 32 pseudonymized clients,
trailing-90-day aggregates. Deliberately **excluded**:
- `trend_direction`, `trend_pct` — the **label source**; never features (asserted in code).
- `content_id`, `client_id` — pseudonymous IDs, grouping/split only.
- `provider_used`, `model_used` — content-gen metadata, not search signals.
- No client names, domains, URLs, or private queries anywhere in `work/`.

Leakage considered: label-derived columns; the `*_last30` comparison windows describe the label
window and are excluded from features; baseline percentiles fit on train only.

## 3. Baseline

The Week-4 transparent rule: `score = stale_flag × visible_flag × visibility_percentile`
(stale ≥ 180 days since update; visible ≥ 500 impressions), with visibility percentile fit on
**train only**. On the held-out test fold it scores precision@50 = **0.32** (it flags very few
high-volume stale pages among unseen clients). A fair comparison because it is scored on the
same data, split, and metric as the model.

## 4. Model / analysis

Gradient-boosting classifier for `P(declining)`, trained on 18 numeric + 8 categorical
(one-hot → 52 columns) leakage-safe features. Fits the lane because the question is a ranked
queue: score every page, order by P(declining), attach a reason code from the observed feature
profile, map to an action. Target (one sentence): *a page is "at risk" if its search
impressions fell sharply (trend_direction = down) in the most recent 30 days.*

## 5. Evaluation

**Split:** grouped by client (80/20, seed 42) → 26 clients (27,675 rows) train, 6 clients
(2,325 rows) test. Every method scored once on this fold. **Metrics (model vs baseline, same
split):** precision@20/50/100 = 0.80 / **0.86** / 0.85, ROC-AUC 0.767, avg precision 0.662;
baseline precision@50 = 0.32; test base rate = 0.39. **Errors:** the model's misses cluster on
mid-volume pages with mediocre position and no extreme feature; false positives are mostly
visibly-high-traffic pages that held despite low CTR — the same region the week-4 signal audit
rated MIXED for CTR-vs-decline.

## 6. Interpretation

Top tree drivers: **days_with_impressions** (reach stability), **content_age_days**, **avg
position** — all observed trailing-90d signals, none label-derived. The top driver is not a
near-perfect predictor (shuffle-importance on AUC ≈ +0.25, not ≈1), so no leakage smell.
Negative/notable result: simple staleness alone is a weak signal; decline concentrates in pages
whose reach is thinning and whose position is mediocre — hence the reason codes around
reach_loss and visible_low_ctr, not just "stale".

## 7. Recommendation

Work the queue top-down: **refresh** stale+visible declining pages and **rewrite** mature
declining pages first; **metadata-review** (title/meta) high-traffic low-CTR pages; **review**
reach-loss / position-loss / high-risk pages; **monitor** the rest. Confidence: high that the
model orders decline risk, **moderate** on the payoff of each action (needs a refresh
experiment). Limits: decision-support, one 90-day slice, six held-out clients, correlational
signals.

## 8. Reproducibility

From a fresh clone:
```bash
git clone https://github.com/himanshu-yadav-10/Flyrank-ML-starter-template
pip install -r requirements.txt
python work/scripts/capstone_analysis.py   # metrics + queue + figures
python work/scripts/build_paper.py          # docs/index.html
```
- **Seeds:** 42 (split + every model). **Environment:** Python 3.11/3.12, scikit-learn 1.6.x,
  pandas 2.2.x, matplotlib 3.x (numbers version-robust for the 0.86 / 0.32 headline).
- **Committed receipts:** `work/outputs/capstone_metrics.json`, `work/outputs/capstone_opportunity_queue.csv`.
- **Notebooks:** `work/notebooks/capstone.ipynb`, `w06_validation_audit.ipynb`,
  `w07_action_playbook.ipynb` (all executed with outputs); weekly building blocks in `work/notebooks/`.

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** (https://flyrank.ai) — a pseudonymized release of
real search-performance data that made this reproducible, honest, applied work possible. No
client-identifying information appears anywhere in this report or the repo.

---

> Claims checklist: observed / measured / directional / decision-support ✓ · base rate reported
> (0.39 test) alongside precision ✓ · no causal refresh-implant claim ✓ · no "predicted Google's
> algorithm" ✓ · numbers match a fresh re-run ✓.
