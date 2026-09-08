"""
Consolidated Cumulative Report Generator Module for Type 2 Diabetes CDSS.

Combines:
1. Patient Profile & Report Dataclasses (PatientDemographics, TrajectoryMetric, RLMonitoringRecommendation, XAIExplanationSummary, CumulativeHealthReport)
2. Multi-Format Report Exporters (JSON, Markdown, Interactive HTML Dashboard)
3. CumulativeReportGenerator Class
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# =========================================================================
# 1. DATA SCHEMAS & DATACLASSES
# =========================================================================

@dataclass
class PatientDemographics:
    subject_id: str
    age: int
    gender: str
    smoking_status: str
    physical_activity: str


@dataclass
class BiomarkerObservation:
    visit_number: int
    chartdate: str
    days_since_last_visit: float
    cumulative_days: float
    hba1c: float
    glucose: float
    bmi: float
    sbp: float
    dbp: float
    cholesterol: float
    ldl: float
    hdl: float
    triglycerides: float
    creatinine: float


@dataclass
class TrajectoryMetric:
    biomarker: str
    current_value: float
    unit: str
    change: float
    rate_monthly: float
    moving_average: float
    trend: str
    clinical_target: str
    is_at_target: bool


@dataclass
class RLMonitoringRecommendation:
    action_code: int
    action_name: str
    recommended_interval: str
    urgency: str
    q_values: Dict[str, float]
    clinical_guidance: str


@dataclass
class XAIExplanationSummary:
    summary_statement: str
    clinical_narrative: str
    key_factors: List[Dict[str, Any]]
    domain_impact: Dict[str, float]


@dataclass
class CumulativeHealthReport:
    report_id: str
    generated_at: str
    demographics: PatientDemographics
    overall_trajectory_status: str
    total_visits: int
    monitoring_duration_days: float
    current_biomarkers: Dict[str, float]
    trajectory_metrics: List[TrajectoryMetric]
    historical_visits: List[Dict[str, Any]]
    rl_recommendation: RLMonitoringRecommendation
    xai_explanation: XAIExplanationSummary

    def to_dict(self) -> Dict[str, Any]:
        import numpy as np
        data = asdict(self)

        def clean_nan(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return 0.0
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(v) for v in obj]
            return obj

        return clean_nan(data)


# =========================================================================
# 2. REPORT GENERATION & EXPORT LOGIC
# =========================================================================

def generate_cumulative_report(
    patient_history_df: pd.DataFrame,
    rl_recommendation_dict: Dict[str, Any],
    xai_explanation_dict: Dict[str, Any],
    report_id: Optional[str] = None,
) -> CumulativeHealthReport:
    """Builds a CumulativeHealthReport instance from longitudinal observations, RL, and XAI."""
    history_sorted = patient_history_df.sort_values(by="chartdate").reset_index(drop=True)
    latest_row = history_sorted.iloc[-1]
    first_row = history_sorted.iloc[0]

    subject_id = str(latest_row.get("subject_id", "Unknown"))
    if report_id is None:
        report_id = f"REP-{subject_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    demographics = PatientDemographics(
        subject_id=subject_id,
        age=int(latest_row.get("age", 60)),
        gender=str(latest_row.get("gender", "Unknown")),
        smoking_status=str(latest_row.get("smoking_status", "never")),
        physical_activity=str(latest_row.get("physical_activity", "moderate")),
    )

    current_biomarkers = {}
    trajectory_metrics = []

    for marker in config.BIOMARKERS:
        curr_val = float(latest_row.get(marker, 0.0))
        current_biomarkers[marker] = curr_val

        delta = float(latest_row.get(f"{marker}_change", 0.0))
        rate_mo = float(latest_row.get(f"{marker}_rate", 0.0))
        ma = float(latest_row.get(f"{marker}_moving_avg", curr_val))
        trend = str(latest_row.get(f"{marker}_trend", "stable"))

        thresh = config.CLINICAL_THRESHOLDS.get(marker, {})
        unit = thresh.get("unit", "")
        target_max = thresh.get("target_max", None)
        target_min = thresh.get("target_min", None)

        if target_max is not None:
            target_str = f"< {target_max} {unit}"
            at_target = curr_val <= target_max
        elif target_min is not None:
            target_str = f">= {target_min} {unit}"
            at_target = curr_val >= target_min
        else:
            target_str = "Normal"
            at_target = True

        trajectory_metrics.append(
            TrajectoryMetric(
                biomarker=marker.upper(),
                current_value=round(curr_val, 2),
                unit=unit,
                change=round(delta, 2),
                rate_monthly=round(rate_mo, 3),
                moving_average=round(ma, 2),
                trend=trend,
                clinical_target=target_str,
                is_at_target=at_target,
            )
        )

    historical_visits = []
    for _, row in history_sorted.iterrows():
        visit_dict = {
            "visit_number": int(row.get("visit_number", 1)),
            "chartdate": str(pd.to_datetime(row["chartdate"]).strftime("%Y-%m-%d")),
            "days_since_last_visit": float(row.get("days_since_last_visit", 0.0)),
            "cumulative_days": float(row.get("cumulative_days", 0.0)),
        }
        for marker in config.BIOMARKERS:
            if marker in row:
                visit_dict[marker] = round(float(row[marker]), 2)
        historical_visits.append(visit_dict)

    duration_days = float(latest_row.get("cumulative_days", 0.0))
    overall_status = str(latest_row.get("overall_trajectory_status", "stable"))

    rl_rec = RLMonitoringRecommendation(
        action_code=int(rl_recommendation_dict.get("action_code", 0)),
        action_name=str(rl_recommendation_dict.get("action_name", "Routine Monitoring")),
        recommended_interval=str(rl_recommendation_dict.get("recommended_interval", "6 months")),
        urgency=str(rl_recommendation_dict.get("urgency", "Low")),
        q_values=rl_recommendation_dict.get("q_values", {}),
        clinical_guidance=str(rl_recommendation_dict.get("clinical_guidance", "")),
    )

    xai_exp = XAIExplanationSummary(
        summary_statement=str(xai_explanation_dict.get("summary_statement", "")),
        clinical_narrative=str(xai_explanation_dict.get("clinical_narrative", "")),
        key_factors=xai_explanation_dict.get("key_factors", xai_explanation_dict.get("key_driving_factors", [])),
        domain_impact=xai_explanation_dict.get("domain_impact", {}),
    )

    return CumulativeHealthReport(
        report_id=report_id,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        demographics=demographics,
        overall_trajectory_status=overall_status,
        total_visits=len(history_sorted),
        monitoring_duration_days=duration_days,
        current_biomarkers=current_biomarkers,
        trajectory_metrics=trajectory_metrics,
        historical_visits=historical_visits,
        rl_recommendation=rl_rec,
        xai_explanation=xai_exp,
    )


def export_report_to_json(report: CumulativeHealthReport, file_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    json_str = json.dumps(report_dict, indent=2)
    with open(file_path, "w") as f:
        f.write(json_str)
    return json_str


def export_report_to_markdown(report: CumulativeHealthReport, file_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    d = report.demographics
    r = report.rl_recommendation
    x = report.xai_explanation

    md_lines = [
        f"# Cumulative Health Report - Type 2 Diabetes CDSS",
        f"**Report ID**: `{report.report_id}` | **Generated**: {report.generated_at}\n",
        f"## 1. Patient Profile",
        f"- **Subject ID**: {d.subject_id}",
        f"- **Age**: {d.age} years | **Gender**: {d.gender}",
        f"- **Smoking Status**: {d.smoking_status.title()} | **Physical Activity**: {d.physical_activity.title()}",
        f"- **Total Follow-up Visits**: {report.total_visits} encounters over {report.monitoring_duration_days:.0f} days",
        f"- **Overall Longitudinal Trajectory**: **{report.overall_trajectory_status.upper()}**\n",
        f"## 2. RL Monitoring & Risk-Management Recommendation",
        f"**Recommended Action**: `{r.action_name}`",
        f"- **Urgency Level**: {r.urgency}",
        f"- **Suggested Monitoring Cadence**: {r.recommended_interval}",
        f"- **Clinical Guidance**: {r.clinical_guidance}\n",
        f"## 3. Explainable AI (SHAP) Clinical Rationale",
        f"> {x.summary_statement}\n",
        f"{x.clinical_narrative}\n",
        f"### Key Contributing Trajectory Indicators:",
    ]

    for factor in x.key_factors[:4]:
        desc = factor.get("clinical_description", factor.get("feature"))
        val = factor.get("shap_attribution", 0.0)
        md_lines.append(f"- **{desc}** (SHAP Attribution: {val:+.4f})")

    md_lines.extend([
        f"\n## 4. Longitudinal Health Trajectory Analysis",
        f"| Biomarker | Current Value | Absolute Change (Δ) | Rate / Month | 3-Visit MA | Clinical Trend | Guideline Target | Target Status |",
        f"|:----------|:-------------:|:-------------------:|:------------:|:----------:|:--------------:|:----------------:|:-------------:|",
    ])

    for tm in report.trajectory_metrics:
        status_icon = "OK On Target" if tm.is_at_target else "⚠ Off Target"
        md_lines.append(
            f"| **{tm.biomarker}** | {tm.current_value} {tm.unit} | {tm.change:+.2f} {tm.unit} | "
            f"{tm.rate_monthly:+.3f} {tm.unit}/mo | {tm.moving_average} {tm.unit} | "
            f"{tm.trend.title()} | {tm.clinical_target} | {status_icon} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return md_content


def export_report_to_html(report: CumulativeHealthReport, file_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    d = report.demographics
    r = report.rl_recommendation
    x = report.xai_explanation

    urgency_color = "#dc3545" if r.urgency == "Critical" else "#fd7e14" if r.urgency == "High" else "#ffc107" if r.urgency == "Moderate" else "#28a745"
    status_badge = "#dc3545" if "deteriorat" in report.overall_trajectory_status else "#28a745" if report.overall_trajectory_status == "improving" else "#17a2b8"

    table_rows = []
    for tm in report.trajectory_metrics:
        target_badge = '<span class="badge badge-success">On Target</span>' if tm.is_at_target else '<span class="badge badge-danger">Off Target</span>'
        trend_class = "text-danger" if tm.trend == "increasing" and tm.biomarker in ["HBA1C", "GLUCOSE", "SBP", "CREATININE"] else "text-success" if tm.trend == "decreasing" else "text-muted"
        table_rows.append(f"""
        <tr>
            <td><strong>{tm.biomarker}</strong></td>
            <td>{tm.current_value} <small>{tm.unit}</small></td>
            <td>{tm.change:+.2f} <small>{tm.unit}</small></td>
            <td>{tm.rate_monthly:+.3f} <small>{tm.unit}/mo</small></td>
            <td>{tm.moving_average} <small>{tm.unit}</small></td>
            <td class="{trend_class}"><strong>{tm.trend.title()}</strong></td>
            <td>{tm.clinical_target}</td>
            <td>{target_badge}</td>
        </tr>
        """)

    factors_html = []
    for f in x.key_factors[:4]:
        factors_html.append(f"""
        <li style="margin-bottom: 8px;">
            <strong>{f.get('clinical_description', f.get('feature'))}</strong>
            <span class="badge" style="background:#e8f0fe; color:#1a73e8; margin-left:6px;">SHAP: {f.get('shap_attribution', 0.0):+.4f}</span>
        </li>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Type 2 Diabetes CDSS - Cumulative Health Report ({d.subject_id})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; color: #212529; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 32px; }}
        .header {{ border-bottom: 2px solid #e9ecef; padding-bottom: 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #1a73e8; }}
        .card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
        .badge {{ display: inline-block; padding: 4px 10px; font-size: 12px; font-weight: 600; border-radius: 4px; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .rec-box {{ border-left: 5px solid {urgency_color}; background: #fff8f8; padding: 18px; border-radius: 6px; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f1f3f4; font-weight: 600; }}
        .text-danger {{ color: #dc3545; }}
        .text-success {{ color: #28a745; }}
        .text-muted {{ color: #6c757d; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div>
            <h1>Type 2 Diabetes Clinical Decision Support Report</h1>
            <small style="color: #6c757d;">Report ID: {report.report_id} | Generated: {report.generated_at}</small>
        </div>
        <span class="badge" style="background:{status_badge}; color:#ffffff; font-size:14px; padding:8px 14px;">
            Trajectory: {report.overall_trajectory_status.upper()}
        </span>
    </div>

    <div class="card" style="display:flex; justify-content:space-between;">
        <div><strong>Patient ID:</strong> {d.subject_id}</div>
        <div><strong>Age:</strong> {d.age}</div>
        <div><strong>Gender:</strong> {d.gender}</div>
        <div><strong>Smoking:</strong> {d.smoking_status.title()}</div>
        <div><strong>Activity:</strong> {d.physical_activity.title()}</div>
        <div><strong>Visits:</strong> {report.total_visits} ({report.monitoring_duration_days:.0f} days)</div>
    </div>

    <div class="rec-box">
        <h3 style="margin-top:0; color:{urgency_color};">Recommended Action: {r.action_name}</h3>
        <p style="margin:4px 0;"><strong>Suggested Cadence:</strong> {r.recommended_interval} | <strong>Urgency:</strong> {r.urgency}</p>
        <p style="margin:4px 0; color:#495057;">{r.clinical_guidance}</p>
    </div>

    <div class="card">
        <h3 style="margin-top:0;">Explainable AI (SHAP) Clinical Rationale</h3>
        <blockquote style="margin:0 0 12px 0; padding-left:12px; border-left:3px solid #1a73e8; color:#1a73e8; font-weight:500;">
            {x.summary_statement}
        </blockquote>
        <p style="color:#495057; font-size:14px;">{x.clinical_narrative}</p>
        <h4 style="margin-bottom:8px;">Key Driving Trajectory Indicators:</h4>
        <ul style="margin:0; padding-left:20px; font-size:14px;">
            {''.join(factors_html)}
        </ul>
    </div>

    <div class="card">
        <h3 style="margin-top:0;">Longitudinal Health Trajectory Analysis</h3>
        <table>
            <thead>
                <tr>
                    <th>Biomarker</th>
                    <th>Current Value</th>
                    <th>Change (Δ)</th>
                    <th>Rate / Month</th>
                    <th>3-Visit MA</th>
                    <th>Trend</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
</div>
</body>
</html>
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_content


def render_markdown_report(report: CumulativeHealthReport) -> str:
    """Renders markdown report directly to string."""
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"tmp_rep_{report.report_id}.md"
    content = export_report_to_markdown(report, tmp)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return content


def render_html_dashboard(report: CumulativeHealthReport) -> str:
    """Renders HTML dashboard directly to string."""
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"tmp_rep_{report.report_id}.html"
    content = export_report_to_html(report, tmp)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return content


class CumulativeReportGenerator:
    """Report Generator service class."""

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or config.REPORTS_DIR

    def generate(
        self,
        patient_history_df: pd.DataFrame,
        rl_recommendation_dict: Dict[str, Any],
        xai_explanation_dict: Dict[str, Any],
        report_id: Optional[str] = None,
        save_formats: Optional[List[str]] = None,
    ) -> CumulativeHealthReport:
        if save_formats is None:
            save_formats = ["json", "md", "html"]

        report = generate_cumulative_report(
            patient_history_df=patient_history_df,
            rl_recommendation_dict=rl_recommendation_dict,
            xai_explanation_dict=xai_explanation_dict,
            report_id=report_id,
        )

        sid = report.demographics.subject_id
        if "json" in save_formats:
            export_report_to_json(report, self.reports_dir / f"{sid}_report.json")
        if "md" in save_formats:
            export_report_to_markdown(report, self.reports_dir / f"{sid}_report.md")
        if "html" in save_formats:
            export_report_to_html(report, self.reports_dir / f"{sid}_report.html")

        return report
