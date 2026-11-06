"""Write results/tables.md and results/chart.png from results/summary.json."""
from __future__ import annotations

import json
from pathlib import Path

# Colours: one highlight hue for edu-v1, neutral gray for everything else.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#2a78d6"
NEUTRAL = "#a3a29c"


def _pct(x) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def _f(x, d=3) -> str:
    return "" if x is None else f"{x:.{d}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def tables_markdown(summary: dict, out_dir: Path) -> str:
    parts = []
    if summary.get("triplets"):
        rows = [[r["model"], r["split"], _pct(r["accuracy"]), _f(r["mean_cos_pos"]), _f(r["mean_cos_neg"]), _f(r["mean_margin"])]
                for r in summary["triplets"]]
        parts.append("### A. Triplet accuracy (private validation splits)\n\n" + _table(
            ["Model", "Split", "Accuracy %", "Mean cos(a,p)", "Mean cos(a,n)", "Mean margin"], rows))
    if summary.get("data_overlap"):
        rows = [[name, _pct(v["corpus_passages_seen_as_training_passages"]),
                 _pct(v["validation_positives_seen_as_training_passages"]),
                 _pct(v["validation_anchors_also_in_training"]), _pct(v["validation_triplets_also_in_training"])]
                for name, v in summary["data_overlap"].items()]
        parts.append("### A. Overlap with the matching training triplets\n\n" + _table(
            ["Split", "Corpus passages seen in training %", "Validation positives seen in training %",
             "Validation anchors in training %", "Validation triplets in training %"], rows))
    if summary.get("cross_section"):
        rows = [[r["model"], r["query"], _pct(r["recall@1"]), _pct(r["recall@5"]), _pct(r["recall@10"]), _f(r["mrr@10"])]
                for r in sorted(summary["cross_section"], key=lambda r: (r["query"] != "plain", -r["recall@10"]))]
        n = summary["cross_section"][0]["n_queries"]
        parts.append(f"### B. Cross-section program retrieval ({n} queries)\n\n" + _table(
            ["Model", "Query format", "R@1 %", "R@5 %", "R@10 %", "MRR@10"], rows))
    if summary.get("interest"):
        rows = [[r["model"], r["setting"], _pct(r["precision@10"]),
                 f"{_pct(r.get('precision@10_ci95_low'))} to {_pct(r.get('precision@10_ci95_high'))}",
                 _pct(r.get("precision_of_returned")), _f(r["returned"], 1), _f(r["unique_unis@10"], 1)]
                for r in summary["interest"]]
        n = summary["interest"][0]["n_queries"]
        parts.append(f"### C. Student-interest queries ({n} queries, author-labelled)\n\n" + _table(
            ["Model", "Setting", "P@10 %", "95% interval", "Precision of returned %", "Programs returned", "Unique universities"], rows))
    health_rows = list(summary.get("health", []))
    prod = out_dir / "health_production_npy.json"
    if prod.exists():
        health_rows.append(json.loads(prod.read_text()))
    if health_rows:
        rows = [[r["model"], str(r["dim"]), _f(r["mean_cos_random_pairs"]), _f(r["std_cos_random_pairs"]), _pct(r["top_pc_variance_share"])]
                for r in health_rows]
        parts.append("### D. Embedding space health (corpus passages)\n\n" + _table(
            ["Embeddings", "Dim", "Mean cos, random pairs", "Std", "Variance on 1st PC %"], rows))
    if summary.get("ablation"):
        models = list(dict.fromkeys(r["model"] for r in summary["ablation"]))
        steps = list(dict.fromkeys(r["step"] for r in summary["ablation"]))
        by = {(r["model"], r["step"]): r for r in summary["ablation"]}
        rows = []
        for s in steps:
            row = [s]
            for m in models:
                r = by.get((m, s)) or {}
                row += [_pct(r.get("precision@10")), _pct(r.get("precision_of_returned")), _f(r.get("returned"), 1)]
            rows.append(row)
        headers = ["Step"] + [h for m in models for h in (f"{m} P@10 %", f"{m} precision of returned %", f"{m} returned")]
        parts.append("### E. Recommender ablation on the student-interest queries\n\n" + _table(headers, rows))
    if summary.get("weight_provenance"):
        rows = []
        for pair, v in summary["weight_provenance"].items():
            blocks = v["relative_diff_by_block"]
            changed = [k for k, d in blocks.items() if d > 0]
            same = v["identical_blocks"]
            desc = ", ".join(changed) if len(changed) <= len(same) else "all except " + ", ".join(same)
            rows.append([pair, _pct(v["overall_relative_diff"]), f"{len(same)} of {len(blocks)}", desc])
        parts.append("### P. Weight comparison with the base model\n\n" + _table(
            ["Pair", "Overall relative difference %", "Identical blocks", "Changed blocks"], rows))
    return "\n\n".join(parts) + "\n"


def _panel(ax, title, labels, values, highlight, ci=None, fmt="{:.1f}"):
    import numpy as np

    y = np.arange(len(labels))[::-1]
    colors = [HIGHLIGHT if lab == highlight else NEUTRAL for lab in labels]
    ax.barh(y, values, height=0.5, color=colors, edgecolor=SURFACE, linewidth=2)
    ends = list(values)
    if ci:
        for yi, (lo, hi) in zip(y, ci):
            ax.plot([lo, hi], [yi, yi], color=TEXT_2, linewidth=1.2, solid_capstyle="butt")
            ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=TEXT_2, linewidth=1.2)
            ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=TEXT_2, linewidth=1.2)
        ends = [max(v, hi) for v, (_, hi) in zip(values, ci)]
    ax.set_yticks(y, labels, fontsize=9.5, color=TEXT)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=TEXT_2, labelsize=8.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for yi, v, e in zip(y, values, ends):
        ax.text(e + 1.5, yi, fmt.format(v), va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_title(title, loc="left", fontsize=10.5, color=TEXT, pad=8)


def write_chart(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    b = [r for r in summary.get("cross_section", []) if r["query"] == "plain"]
    if b:
        b.sort(key=lambda r: -r["recall@10"])
        panels.append(("B. Cross-section program retrieval, Recall@10 % (3,695 queries)",
                       [r["model"] for r in b], [100 * r["recall@10"] for r in b], None))
    c = [r for r in summary.get("interest", []) if r["setting"] == "dense, plain query"]
    if c:
        c.sort(key=lambda r: -r["precision@10"])
        ci = [(100 * r["precision@10_ci95_low"], 100 * r["precision@10_ci95_high"]) for r in c] \
            if all("precision@10_ci95_low" in r for r in c) else None
        panels.append(("C. Student-interest queries, P@10 %, dense only (40 queries, 95% interval)",
                       [r["model"] for r in c], [100 * r["precision@10"] for r in c], ci))
    if not panels:
        return
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    heights = [len(p[1]) * 0.42 + 0.9 for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, sum(heights)), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": heights})
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, labels, values, ci) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        _panel(ax, title, labels, values, highlight="edu-v1", ci=ci)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict) -> None:
    out_dir = Path(out_dir)
    (out_dir / "tables.md").write_text(tables_markdown(summary, out_dir), encoding="utf-8")
    write_chart(summary, out_dir / "chart.png")
