# Template-disjoint splits (replication artifact)

Five 70/30 splits of the 8,864-instance pool grouped by BBQ template
`(category, question_index)` via `GroupShuffleSplit(random_state=seed)`,
so train and test share **no template**. These are the exact splits behind
the paper's template-disjoint control (embedding-only condition accuracy
0.8874 +- 0.0369; condition-only retention 0.9319 / 0.7993 / 0.1480).

`uid = "<Category>::<example_id>"` indexes `data/sampled_v2`.
Regenerate deterministically with `scripts/export_template_disjoint_splits.py`.
