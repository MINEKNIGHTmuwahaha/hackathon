"""
src/report.py
-------------
Generate a JSON report + optional PDF summary from metrics.
"""

import json
from pathlib import Path
from datetime import datetime

from .logger import get_logger

log = get_logger("sentinel.report")


def save_metrics_json(metrics: dict, out_path: str = "docs/metrics.json") -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.utcnow().isoformat(), **metrics}
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info(f"metrics -> {p}")
    return str(p)


def save_summary_markdown(metrics: dict, out_path: str = "docs/SUMMARY.md") -> str:
    lines = [
        "# Sentinel ECG — Evaluation Summary",
        f"_Generated: {datetime.utcnow().isoformat()}_",
        "",
        f"**Accuracy:** {metrics.get('accuracy', 'n/a')}",
        f"**Referral rate:** {metrics.get('referral_rate', 'n/a')}",
        "",
        "## Per-class recall",
    ]
    for cls, val in (metrics.get("per_class_recall") or {}).items():
        lines.append(f"- **{cls}**: {val}")
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("```")
    lines.append(str(metrics.get("confusion_matrix", "")))
    lines.append("```")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"summary -> {p}")
    return str(p)


def try_pdf(summary_md_path: str, pdf_path: str = "docs/summary.pdf") -> str | None:
    """
    Try to render Markdown to PDF via fpdf2 (optional).
    Returns path or None if unavailable.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        log.warning("fpdf2 not installed — skipping PDF")
        return None

    text = Path(summary_md_path).read_text(encoding="utf-8")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line)
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(pdf_path)
    log.info(f"pdf -> {pdf_path}")
    return pdf_path