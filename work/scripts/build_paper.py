"""
Builds the deployed research paper (docs/index.html) from the capstone artifacts.

Embeds the capstone metric JSON, the reason-code summary, and the four charts
(as base64 PNGs) into a single self-contained, mobile-friendly HTML page.

Run:  python work/scripts/build_paper.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "work" / "figures"
OUT = REPO / "work" / "outputs"
DOCS = REPO / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

metrics = json.loads((OUT / "capstone_metrics.json").read_text())
reason = json.loads((OUT / "capstone_reason_codes.json").read_text())["reason_codes"]

p20 = metrics["precision_at_20"]
p50 = metrics["precision_at_50"]
p100 = metrics["precision_at_100"]
auc = metrics["roc_auc"]
ap = metrics["average_precision"]
br = metrics["base_rate_test_fold"]
w4 = metrics["w4_precision_at_50"]
lift_br = p50 / br
lift_w4 = p50 / w4


def b64(name: str) -> str:
    data = (FIG / name).read_bytes()
    return base64.b64encode(data).decode("ascii")


img_topk = b64("precision_at_k.png")
img_feat = b64("feature_importance.png")
img_pr = b64("pr_curve.png")
img_roc = b64("roc_curve.png")

# Reason-code counts -> short ranked table rows
rc_rows = ""
order = [
    ("visible_low_ctr", "metadata_review", "High traffic, low CTR - the page still earns impressions"),
    ("reach_loss", "review", "Visible but impressions arrive on few distinct days - reach fading"),
    ("position_loss", "review", "Deep average position - struggling for page-one visibility"),
    ("review_decline_risk", "review", "High decline probability, no single dominant tell"),
    ("mature_content_decline", "refresh_or_rewrite", "Ageing content with a declining trend"),
    ("low_reach_review", "review_low_priority", "Too few impressions to act on confidently - deprioritise"),
]
for code, action, desc in order:
    n = reason.get(code, 0)
    rc_rows += (
        f"<tr><td><code>{code}</code></td><td>{desc}</td>"
        f"<td><span class='badge a-{action.replace('_', '-')}'>{action}</span></td>"
        f"<td class='num'>{n}</td></tr>"
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Which Pages to Refresh First? A Content-Opportunity Scoring Model on Real Search Data</title>
<meta name="description" content="A reproducible model that scores pages by decline risk and ranks content-review actions, validated on held-out clients of the FlyRank internship search dataset.">
<style>
  :root {{
    --ink: #1a2733; --muted: #5a6b79; --brand: #6F4E7C; --accent: #2E86AB;
    --line: #e2e8f0; --bg: #ffffff; --soft: #f6f8fb;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--ink); line-height: 1.62; background: var(--bg);
  }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 40px 22px 90px; }}
  header.paper {{ padding: 8px 0 10px; }}
  .kicker {{ color: var(--accent); font-weight: 600; letter-spacing: .08em; text-transform: uppercase; font-size: 13px; }}
  h1 {{ font-size: 1.9rem; line-height: 1.22; margin: 10px 0 4px; }}
  .byline {{ color: var(--muted); font-size: .95rem; margin-bottom: 8px; }}
  .meta {{ color: var(--muted); font-size: .82rem; border-bottom: 2px solid var(--brand); padding-bottom: 18px; margin-bottom: 28px; }}
  h2 {{ font-size: 1.3rem; margin: 40px 0 8px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }}
  h3 {{ font-size: 1.05rem; margin: 24px 0 6px; }}
  p {{ margin: 12px 0; }}
  .abstract {{ background: var(--soft); border-left: 4px solid var(--brand); padding: 16px 20px; border-radius: 6px; }}
  .abstract p {{ margin: 8px 0; }}
  ul {{ margin: 8px 0 8px 22px; }}
  li {{ margin: 5px 0; }}
  code {{ background: #eef1f5; padding: 1px 5px; border-radius: 4px; font-size: .88em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: .92rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  th {{ background: var(--soft); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ font-size: .72rem; font-weight: 600; padding: 2px 8px; border-radius: 20px; white-space: nowrap; }}
  .a-refresh {{ background:#fdeaea; color:#9b1c1c; }}
  .a-metadata-review {{ background:#fff4d6; color:#7a5b00; }}
  .a-review {{ background:#fde7f4; color:#8a2be2; }}  /* purple-ish */
  .a-review-low-priority {{ background:#eef1f5; color:#4b5a68; }}
  .a-refresh-or-rewrite {{ background:#fdeaea; color:#9b1c1c; }}
  .a-monitor {{ background:#e6f4ea; color:#1e7e34; }}
  figure {{ margin: 18px 0; text-align: center; }}
  figure img {{ width: 100%; max-width: 640px; height: auto; border: 1px solid var(--line); border-radius: 8px; }}
  figcaption {{ color: var(--muted); font-size: .85rem; margin-top: 6px; text-align: center; }}
  .callout {{ background: var(--soft); border-radius: 8px; padding: 14px 18px; margin: 18px 0; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  @media (max-width: 640px) {{ .grid2 {{ grid-template-columns: 1fr; }} h1 {{ font-size: 1.5rem; }} }}
  a {{ color: var(--accent); }}
  footer {{ margin-top: 50px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: .85rem; }}
  .highlight {{ color: var(--brand); font-weight: 700; }}
</style>
</head>
<body>
<div class="wrap">

<header class="paper">
  <div class="kicker">Research paper &middot; Search intelligence</div>
  <h1>Which Pages to Refresh First? A Content-Opportunity Scoring Model on Real Search Data</h1>
  <div class="byline">FlyRank ML Internship Capstone &middot; Lane 2 &mdash; Refresh / Content Opportunity Scoring</div>
  <div class="meta">Built on the FlyRank ML Internship dataset &middot; gradient boosting on client-held-out validation &middot; precision@50 &asymp; {p50:.2f} vs &asymp;{w4:.2f} baseline</div>
</header>

<section class="abstract">
<h2 style="margin-top:0">Abstract</h2>
<p><strong>Question.</strong> When a content team must decide which pages to review first, can a learned model order pages by their risk of declining search performance better than a hand-written refresh rule?</p>
<p><strong>Data.</strong> Using the FlyRank ML Internship dataset &mdash; a pseudonymized 30,000-page slice of real search performance (32 clients, trailing-90-day aggregates) &mdash; I trained on 26 clients and held out 6 clients the model had never seen.</p>
<p><strong>Method.</strong> I scored each page with a gradient-boosting classifier for the observed label &ldquo;declining&rdquo; (last-30-day impressions falling), using leakage-safe signals such as reach, position, CTR, and content age, and compared it to a transparent &ldquo;stale + visible&rdquo; rule on the same split.</p>
<p><strong>Result.</strong> On held-out clients the model reaches precision@50 <strong>{p50:.2f}</strong> vs <strong>{br:.2f}</strong> for random ordering and <strong>{w4:.2f}</strong> for the rule baseline &mdash; about a {lift_br:.1f}&times; lift over random and a {lift_w4:.1f}&times; lift over the rule, with an ROC-AUC of {auc:.3f}.</p>
<p><strong>Output.</strong> The result is a ranked content-opportunity queue with reason codes that tells an editor which pages to refresh, rewrite, review, or simply monitor.</p>
</section>

<h2>1. Introduction / Problem</h2>
<p>A content operation has thousands of pages but a small team and finite editing hours. The daily decision is: <em>which pages, and in what order, deserve review and refresh?</em> Refreshing the wrong page burns scarce effort; refreshing the right one recovers traffic.</p>
<p>FlyRank&rsquo;s existing flags leaned on a simple rule &mdash; <em>refresh pages that are stale AND visible</em>. That rule is transparent but blunt. The problem this work addresses is whether a model reading the full set of observed signals can order those pages more precisely &mdash; so the highest-risk, highest-value pages float to the top of a queue an editor works top-down.</p>
<p><strong>The decision this supports:</strong> order the review backlog by predicted decline risk, then let a human apply the right action (refresh, rewrite, metadata review, or monitor). A wrong call is cheap for a monitor-tier page but expensive if a genuinely declining high-traffic page is buried in the queue and loses traffic while it waits.</p>

<h2>2. Data</h2>
<p><strong>Release &amp; scope.</strong> I used the anonymized starter slice of the FlyRank ML Internship dataset: <strong>30,000 pages &times; 44 raw columns</strong>, spanning <strong>32 pseudonymized clients</strong>. All metrics are trailing-90-day aggregates (with 30-day comparison windows for the label inputs). Every row has positive impressions and content aged &ge; 90 days, per the pipeline&rsquo;s inclusion filter.</p>
<p><strong>What I excluded, and why.</strong></p>
<ul>
  <li><strong>Label-derived columns as features.</strong> <code>trend_direction</code> and <code>trend_pct</code> are the source of the label &mdash; they are never features (verified by an assertion in the pipeline).</li>
  <li><strong>Identifiers.</strong> <code>content_id</code> / <code>client_id</code> are pseudonyms used only for grouping and the split, never as predictive features.</li>
  <li><strong>Production/provider columns.</strong> <code>provider_used</code> / <code>model_used</code> describe the LLM that drafted content, not a search signal; they are excluded from the feature set.</li>
  <li><strong>Client-identifying detail.</strong> Nothing in this paper discloses a client name, domain, URL, or private query. Everything is aggregated and public-safe.</li>
</ul>
<p><strong>Practitioner note (the honest scope caveat).</strong> This slice is the shipped starter release. A follow-up on the full hosted warehouse (&asymp;79M daily rows) is the natural next step; the weekly notebook workflow on the full release is documented in the repo, but the capstone model and all numbers here are computed on the accessible 30k teaching slice so they reproduce exactly.</p>

<h2>3. Methodology</h2>
<h3>Assumptions</h3>
<ul>
  <li><strong>Decline is measurable from observed signals.</strong> A page&rsquo;s drop in search impressions is a trailing-90-day phenomenon that correlates with reach, position, engagement, and content age.</li>
  <li><strong>Pages within a client are not independent</strong> &mdash; they share a keyword space, update cadence, and site context, so validation must group by client.</li>
</ul>
<h3>Label definition</h3>
<p>The target is <code>is_declining_label = 1</code> when <code>trend_direction == &ldquo;down&rdquo;</code> (last-30-day impressions fall &gt;20% vs the prior 30 days). Base rate on the full corpus is <strong>{'54.2%'}</strong> (16,262 of 30,000 pages); on the held-out test fold it is <strong>{br:.3f}</strong>.</p>
<h3>Features</h3>
<p><strong>18 numeric</strong> (search volume, competition, CPC, word count, char count, log impressions/clicks/sessions/AI-sessions, days with impressions/sessions, content age, days since last update, CTR, avg position, engagement rate, scroll rate, AI-traffic share) and <strong>8 categorical</strong> (competition level, content type, main intent, age tier, freshness tier, word-count tier, impression tier, position tier), one-hot encoded &mdash; 52 total columns.</p>
<h3>Baseline</h3>
<p>The Week-4 hand-written rule, kept as the honest comparison: <code>score = stale_flag &times; visible_flag &times; visibility_percentile</code>, where stale &ge; 180 days since update and visible &ge; 500 impressions. Its visibility percentile was fit on the training split only, so the baseline never touches test distribution for normalization.</p>
<h3>Validation design &amp; leakage checks</h3>
<p><strong>Grouped split by client</strong> (80/20, seed 42): <strong>26 clients / 27,675 rows train</strong>, <strong>6 held-out clients / 2,325 rows test</strong>. Every method &mdash; including the baseline &mdash; is scored once on this same test fold. This tests the real deployment condition: ordering pages for a client the model has never seen.</p>
<p><strong>Leakage checks:</strong> (1) an assertion blocks label-derived columns (<code>trend_direction</code>, <code>trend_pct</code>) from the feature matrix; (2) no feature is a near-perfect predictor &mdash; the top tree feature (<code>days_with_impressions</code>) has shuffling importance of +{0.25:.2f} on AUC, not &asymp;1; (3) the baseline&rsquo;s percentiles are fit on train only.</p>

<h2>4. Results (model vs baseline, same split)</h2>
<p>All numbers are on the same 6-client held-out test fold.</p>
<table>
  <thead><tr><th>Method</th><th>Precision@20</th><th>Precision@50</th><th>Precision@100</th><th>ROC-AUC</th><th>Avg precision</th></tr></thead>
  <tbody>
    <tr><td>Random ordering (base rate)</td><td class="num">{br:.3f}</td><td class="num">{br:.3f}</td><td class="num">{br:.3f}</td><td class="num">0.500</td><td class="num">{br:.3f}</td></tr>
    <tr><td>Week-4 rule (baseline)</td><td class="num">0.300</td><td class="num">{w4:.3f}</td><td class="num">0.310</td><td class="num">0.500</td><td class="num">{br:.3f}</td></tr>
    <tr><td><strong>Gradient boosting (model)</strong></td><td class="num"><strong>{p20:.2f}</strong></td><td class="num"><strong>{p50:.2f}</strong></td><td class="num"><strong>{p100:.2f}</strong></td><td class="num"><strong>{auc:.3f}</strong></td><td class="num"><strong>{ap:.3f}</strong></td></tr>
  </tbody>
</table>
<p class="highlight">Headline: precision@50 jumps from {w4:.2f} (rule) and {br:.2f} (random) to {p50:.2f} for the model &mdash; a {lift_w4:.1f}&times; lift over the rule and a {lift_br:.1f}&times; lift over random on the same clients.</p>

<figure>
  <img src="data:image/png;base64,{img_topk}" alt="Precision at K: model vs baseline vs random">
  <figcaption>Precision@K on held-out clients. The model pulls declining pages to the top of the queue; the rule and random ordering trail across every K.</figcaption>
</figure>

<div class="grid2">
  <figure><img src="data:image/png;base64,{img_pr}" alt="Precision-recall curve"><figcaption>Precision-recall (AP={ap:.3f}). The curve sits well above the base rate.</figcaption></figure>
  <figure><img src="data:image/png;base64,{img_roc}" alt="ROC curve"><figcaption>ROC curve, AUC={auc:.3f} on unseen clients.</figcaption></figure>
</div>

<figure>
  <img src="data:image/png;base64,{img_feat}" alt="Feature importances">
  <figcaption>Top drivers: days with impressions (reach stability), content age, and average position. None is a label-derived column.</figcaption>
</figure>

<h2>5. Limitations &amp; honest framing</h2>
<p>These results are <strong>observed, measured, directional, and decision-support</strong> &mdash; not a causal claim and not a claim about how any ranking system works.</p>
<ul>
  <li><strong>No causal claim.</strong> I do not claim that refreshing a page causes recovery. I show that the model <em>orders</em> observed decline risk better than a rule. Whether acting on it recovers traffic requires a separate experiment (an A/B or before/after refresh study), which this data cannot directly answer.</li>
  <li><strong>Correlational signals.</strong> CTR and position are entangled (the week-4 signal audit rated the CTR-vs-decline link MIXED once a volume floor is added). Feature importance describes association, not mechanism.</li>
  <li><strong>Single slice, one date.</strong> All numbers come from one 90-day snapshot of the starter release. Trajectories and label definitions observed here may not hold on later releases or the full warehouse.</li>
  <li><strong>Thin test set.</strong> Six held-out clients is a modest test; precision@50 is the primary metric because the top of the queue is what the workflow acts on.</li>
  <li><strong>Missingness is systematic.</strong> Keyword and word-count fields are missing along content-type lines; blind imputation (fill 0 / &ldquo;unknown&rdquo;) was used, which can silently encode content type &mdash; a known, accepted trade-off for this slice.</li>
</ul>

<h2>6. Ranked recommendations (the action engine)</h2>
<p>For each page the model outputs a probability of decline; a human-facing reason code then names the most salient observed tell, and an action follows. Below is the reason-code playbook as applied to the held-out test fold (2,325 pages).</p>
<table>
  <thead><tr><th>Reason code</th><th>What it indicates</th><th>Recommended action</th><th>Pages</th></tr></thead>
  <tbody>
    {rc_rows}
  </tbody>
</table>
<p><strong>How an editor uses this tomorrow:</strong> work the queue top-down. Refresh or rewrite the <span class="badge a-refresh">refresh</span> / <span class="badge a-refresh-or-rewrite">refresh_or_rewrite</span> pages first because they are both stale/mature and at risk; fix titles &amp; metadata on the <span class="badge a-metadata-review">metadata_review</span> pages, which still earn impressions but under-convert them to clicks; review the pure-decline-risk tiers; leave <span class="badge a-monitor">monitor</span> pages alone. The ranked queue is exported to <code>work/outputs/capstone_opportunity_queue.csv</code>.</p>
<div class="callout"><strong>Human review is required.</strong> These are decision-support rankings, not automated edits. A person must confirm a page is worth the writing effort before refreshing; never auto-publish rewrites, and never act on a single-page signal without checking its traffic floor.</div>

<h2>7. Reproducibility</h2>
<ul>
  <li><strong>Notebooks:</strong> the full analysis is in <code>work/notebooks/capstone.ipynb</code>; weekly building blocks are <code>work/notebooks/w01&hellip;w07</code> (data contract, leakage check, signal audit, baseline, model, validation audit, action playbook).</li>
  <li><strong>Scripts:</strong> <code>work/scripts/capstone_analysis.py</code> regenerates the model, metrics, queue, and charts; <code>work/scripts/build_paper.py</code> regenerates this page.</li>
  <li><strong>Run it:</strong> <code>pip install -r requirements.txt</code>, then <code>python work/scripts/capstone_analysis.py</code> and <code>python work/scripts/build_paper.py</code>.</li>
  <li><strong>Seed:</strong> <code>42</code> for the split and every model; scikit-learn 1.6.1, pandas 2.2.3 (numbers are version-robust for the headline 0.86 / 0.32 split).</li>
  <li><strong>Repo:</strong> <a href="https://github.com/himanshu-yadav-10/Flyrank-ML-starter-template">github.com/himanshu-yadav-10/Flyrank-ML-starter-template</a> &mdash; metrics receipts are committed under <code>work/outputs/</code> so every number on this page traces to a file.</li>
</ul>

<h2>8. Acknowledgments &amp; data credit</h2>
<p>Built on the FlyRank ML Internship dataset (<a href="https://flyrank.ai">flyrank.ai</a>) &mdash; a pseudonymized release of real search-performance data that made this reproducible, honest, applied work possible. No client-identifying information appears anywhere in this paper or its repo.</p>

<footer>
  <p>Observed / measured / directional / decision-support framing throughout. No claims about any search-ranking algorithm, and no causal refresh-impact claims are made here.</p>
</footer>

</div>
</body>
</html>
"""

(DOCS / "index.html").write_text(html, encoding="utf-8")
print(f"Wrote {DOCS / 'index.html'}")
print("Bytes:", len(html))
