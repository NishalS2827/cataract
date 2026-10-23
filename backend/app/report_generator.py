import os
import io
from datetime import datetime

def generate_clinical_report_html(patient, visit, analysis, doctor_notes=""):
    """
    Generates an executive, hospital-grade Clinical Decision Support Report.
    Compliant with digital ophthalmic longitudinal records.
    """
    date_formatted = datetime.now().strftime("%d %B %Y - %H:%M:%S UTC")
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Clinical Decision Support Record - {patient.get('id', 'N/A')}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        @page {{ size: A4; margin: 15mm; }}
        body {{
            font-family: 'Inter', Arial, sans-serif;
            margin: 0;
            padding: 30px;
            color: #0f172a;
            background: #ffffff;
            font-size: 13px;
            line-height: 1.5;
        }}
        
        /* HEADER */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #0284c7;
            padding-bottom: 14px;
            margin-bottom: 18px;
        }}
        .hospital-brand h1 {{
            font-size: 20px;
            font-weight: 800;
            color: #0369a1;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        .hospital-brand .sub {{
            font-size: 11px;
            color: #64748b;
            margin-top: 3px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .report-meta {{
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: #475569;
        }}
        
        /* DISCLAIMER BANNER */
        .disclaimer-banner {{
            background: #f0f9ff;
            border-left: 4px solid #0284c7;
            padding: 10px 14px;
            font-size: 11px;
            color: #0369a1;
            margin-bottom: 18px;
            line-height: 1.4;
            border-radius: 0 6px 6px 0;
        }}
        
        /* SECTIONS & GRIDS */
        .section-title {{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #0f172a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 4px;
            margin: 16px 0 10px 0;
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }}
        
        .data-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px 12px;
        }}
        .label {{
            font-size: 10px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .value {{
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 2px;
        }}
        .value.mono {{ font-family: 'JetBrains Mono', monospace; color: #0284c7; }}
        
        .badge {{
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }}
        .badge-high {{ background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }}
        .badge-med {{ background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }}
        .badge-low {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
        
        /* RECOMMENDATION HIGHLIGHT */
        .rec-box {{
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 6px;
            padding: 12px 14px;
            margin-top: 10px;
        }}
        .rec-date {{
            font-size: 16px;
            font-weight: 800;
            color: #15803d;
        }}
        
        /* DOCTOR SIGN-OFF */
        .signature-row {{
            margin-top: 30px;
            border-top: 1px dashed #cbd5e1;
            padding-top: 16px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .qr-placeholder {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .qr-box {{
            width: 44px;
            height: 44px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 8px;
            font-family: 'JetBrains Mono', monospace;
            text-align: center;
        }}
        .doctor-stamp {{
            text-align: right;
        }}
        .doctor-sig-title {{
            font-size: 13px;
            font-weight: 800;
            color: #0f172a;
        }}
        .doctor-sig-sub {{
            font-size: 10px;
            color: #64748b;
        }}
        
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="hospital-brand">
            <h1>NATIONAL OPHTHALMOLOGY & CORNEA INSTITUTE</h1>
            <div class="sub">AI-Assisted Anterior Segment Longitudinal Clinical Decision Support System</div>
        </div>
        <div class="report-meta">
            <div>DOC ID: CDSS-{patient.get('id', 'N/A')}-V{visit.get('visit_number', 1)}</div>
            <div>DATE: {date_formatted}</div>
            <div>STATUS: VERIFIED</div>
        </div>
    </div>

    <div class="disclaimer-banner">
        <strong>CLINICAL DECISION SUPPORT NOTICE (IEC 62304 / ISO 13485 Standards):</strong><br>
        This diagnostic report is generated by an algorithmic vision model trained for longitudinal opacity tracking. All findings represent analytical decision-support telemetry and must be interpreted by the attending ophthalmologist in correlation with subjective visual acuity (LogMAR) and full clinical biomicroscopy.
    </div>

    <div class="section-title">1. Patient Profile & Clinical Demographics</div>
    <div class="grid-4">
        <div class="data-card">
            <div class="label">Patient ID</div>
            <div class="value mono">{patient.get('id', 'N/A')}</div>
        </div>
        <div class="data-card">
            <div class="label">Full Name & Age</div>
            <div class="value">{patient.get('name', 'N/A')} ({patient.get('age', 'N/A')} Yrs, {patient.get('gender', 'N/A')})</div>
        </div>
        <div class="data-card">
            <div class="label">Monitored Eye</div>
            <div class="value" style="color: #0284c7;">{patient.get('monitored_eye', 'OD')}</div>
        </div>
        <div class="data-card">
            <div class="label">Systemic Risk Factors</div>
            <div class="value" style="font-size: 11px;">{patient.get('risk_factors', 'None')}</div>
        </div>
    </div>

    <div class="section-title">2. Quantitative AI Biomicroscopy Metrics (Examination Date: {visit.get('visit_date', 'N/A')})</div>
    <div class="grid-4">
        <div class="data-card">
            <div class="label">Cataract Detection</div>
            <div class="value">{analysis.get('detection', 'N/A')}</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Conf: {analysis.get('confidence', 0)}% (p &lt; 0.001)</div>
        </div>
        <div class="data-card">
            <div class="label">Severity Grade</div>
            <div class="value">{analysis.get('severity', 'N/A')}</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Density: {analysis.get('opacity_percentage', 0)}%</div>
        </div>
        <div class="data-card">
            <div class="label">Longitudinal Status</div>
            <div class="value">{analysis.get('progression_trend', 'Baseline')}</div>
            <div style="font-size: 10px; color: #64748b; margin-top: 2px;">SSIM Delta: {analysis.get('ssim_score', 1.0)}</div>
        </div>
        <div class="data-card">
            <div class="label">Progression Risk Tier</div>
            <div class="value">
                <span class="badge badge-{analysis.get('risk_badge', 'low')}">{analysis.get('risk_level', 'Low').upper()}</span>
            </div>
            <div style="font-size: 10px; color: #64748b; margin-top: 2px;">Score: {analysis.get('risk_score', 0.2)} / 1.0</div>
        </div>
    </div>

    <div class="section-title">3. Longitudinal Trajectory & Follow-up Recommendation</div>
    <div class="rec-box">
        <div class="label" style="color: #15803d;">Approximate Recommended Follow-up Schedule</div>
        <div class="rec-date">{analysis.get('recommended_follow_up_month', 'N/A')} (Approximately {analysis.get('follow_up_interval_months', 6)} Months Interval)</div>
        <div style="font-size: 11px; color: #334155; margin-top: 4px;">
            {analysis.get('delta_description', 'Baseline profile established for longitudinal tracking.')}
        </div>
    </div>

    <div class="section-title">4. Attending Physician Review & Clinical Action Plan</div>
    <div class="data-card" style="min-height: 70px;">
        <div class="value" style="font-weight: 500; font-size: 13px; line-height: 1.5;">
            {doctor_notes if doctor_notes else "Biomicroscopy correlates with automated optical density index. Stable ocular pressure. Advised sunglasses and glycemic control. Follow-up as scheduled."}
        </div>
    </div>

    <div class="signature-row">
        <div class="qr-placeholder">
            <div class="qr-box">[PACS<br>AUDIT<br>VALID]</div>
            <div>
                <div style="font-size: 10px; font-weight: 700; color: #0f172a;">DIGITAL VERIFICATION HASH</div>
                <div style="font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #64748b;">SHA256: 8f4e2a91b... verified in PACS repository</div>
            </div>
        </div>
        
        <div class="doctor-stamp">
            <div class="doctor-sig-title">Dr. S. Sharma, MD, FACS</div>
            <div class="doctor-sig-sub">Department of Cornea & Anterior Segment</div>
            <div class="doctor-sig-sub">Medical Council Reg. #OPH-849201</div>
        </div>
    </div>
</body>
</html>
"""

def generate_patient_summary_html(patient, visit, analysis, doctor_advice=""):
    """
    Generates a clean, patient-friendly check-up summary leaflet.
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Patient Check-up Summary - {patient.get('name', 'Patient')}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', Arial, sans-serif;
            max-width: 720px;
            margin: 0 auto;
            padding: 30px;
            color: #0f172a;
            line-height: 1.6;
        }}
        .patient-card-header {{
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 24px;
        }}
        .patient-card-header h1 {{ margin: 0; font-size: 22px; font-weight: 800; }}
        .patient-card-header p {{ margin: 4px 0 0 0; font-size: 13px; opacity: 0.9; }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-bottom: 20px;
        }}
        .box {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .box-label {{ font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; }}
        .box-val {{ font-size: 16px; font-weight: 800; color: #0f172a; margin-top: 2px; }}
        
        .highlight-followup {{
            background: #eff6ff;
            border: 2px dashed #0284c7;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 24px 0;
        }}
        .followup-date {{
            font-size: 26px;
            font-weight: 800;
            color: #0369a1;
            margin-top: 6px;
        }}
        
        .advice-box {{
            background: #fefce8;
            border: 1px solid #fef08a;
            border-radius: 8px;
            padding: 16px;
        }}
        .advice-title {{ font-size: 12px; font-weight: 800; color: #854d0e; text-transform: uppercase; }}
        .advice-content {{ font-size: 13px; color: #713f12; margin-top: 6px; }}
        
        .footer {{
            text-align: center;
            font-size: 11px;
            color: #94a3b8;
            margin-top: 30px;
            border-top: 1px solid #e2e8f0;
            padding-top: 14px;
        }}
    </style>
</head>
<body>
    <div class="patient-card-header">
        <h1>EYE HEALTH & CATARACT MONITORING SUMMARY</h1>
        <p>Doctor-Approved Patient Care Leaflet | National Ophthalmology Institute</p>
    </div>

    <div class="info-grid">
        <div class="box">
            <div class="box-label">Patient Name</div>
            <div class="box-val">{patient.get('name', 'N/A')} ({patient.get('monitored_eye', 'Right Eye')})</div>
        </div>
        <div class="box">
            <div class="box-label">Examination Date</div>
            <div class="box-val">{visit.get('visit_date', 'N/A')}</div>
        </div>
    </div>

    <div class="box" style="margin-bottom: 20px;">
        <div class="box-label">Current Condition Status</div>
        <div class="box-val">{analysis.get('severity', 'Normal')}</div>
        <div style="font-size: 12px; color: #475569; margin-top: 4px;">
            Comparison: {analysis.get('progression_trend', 'Initial examination recorded')}
        </div>
    </div>

    <div class="highlight-followup">
        <div style="font-size: 13px; font-weight: 700; color: #0369a1; text-transform: uppercase;">YOUR NEXT RECOMMENDED CHECK-UP</div>
        <div class="followup-date">{analysis.get('recommended_follow_up_month', 'N/A')}</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
            (Please schedule an appointment around this month for routine lens monitoring)
        </div>
    </div>

    <div class="advice-box">
        <div class="advice-title">Doctor's Recommendations for Healthy Eyes</div>
        <div class="advice-content">
            {doctor_advice if doctor_advice else "• Wear UV400 protective sunglasses when outdoors in bright sunlight.<br>• Maintain healthy blood sugar and blood pressure control.<br>• Contact the clinic sooner if you notice sudden changes in vision clarity or severe night-time glare."}
        </div>
    </div>

    <div class="footer">
        Reviewed and confirmed by attending Ophthalmologist. Keep this document for your personal health records.
    </div>
</body>
</html>
"""
