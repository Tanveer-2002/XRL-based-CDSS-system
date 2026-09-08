"""Cumulative Health Report Generator (Re-exported from consolidated report/generator)."""
from .generator import (
    generate_cumulative_report,
    export_report_to_json,
    export_report_to_markdown,
    export_report_to_html,
    render_html_dashboard,
    render_markdown_report,
    CumulativeReportGenerator,
)

__all__ = [
    "generate_cumulative_report",
    "export_report_to_json",
    "export_report_to_markdown",
    "export_report_to_html",
    "render_html_dashboard",
    "render_markdown_report",
    "CumulativeReportGenerator",
]
