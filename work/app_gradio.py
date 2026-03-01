#!/usr/bin/env python3
"""
UNBOXED — Interface Gradio pour le pipeline radiologique agentique.
Hackathon GE Healthcare x Centrale Lyon 2026.

Usage:
    python app_gradio.py
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ── Setup paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent
EXCEL_PATH = BASE_DIR / "Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx"
PATIENTS_DIR = SCRIPT_DIR / "patients"

# Add work/ to path for pipeline imports
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    load_gemini_key,
    get_orthanc_session,
    step_discovery,
    step_download,
    step_segmentation,
    step_analyse_seg,
    step_clinical_context,
    classify_findings,
    build_lesion_tracking_table,
    build_recist_evaluation,
    extract_exam_motif,
    compute_confidence,
    compute_all_vdt,
    format_vdt_text,
    format_recist_reasoning,
    format_lungrads_reasoning,
    analyze_hounsfield,
    audit_data_completeness,
    format_tagged_data,
    build_sources_trace,
    step_generate_report,
    verify_report,
    build_correction_prompt,
    call_gemini_correction,
    evaluate_report_quality,
    step_output,
    classify_clinical_context,
)
import pydicom


# ── Load patients from Excel ──────────────────────────────────────
def load_patient_choices():
    """Build dropdown choices — only accessions with local DICOM data or results."""
    if not EXCEL_PATH.exists():
        return ["Excel non trouvé — vérifiez le chemin"]
    df = pd.read_excel(str(EXCEL_PATH))

    # Check Orthanc availability once
    orthanc_ok = False
    try:
        get_orthanc_session()
        orthanc_ok = True
    except Exception:
        pass

    choices = []
    for pid, group in df.groupby("PatientID"):
        accs = sorted(group["AccessionNumber"].astype(str).tolist())
        n = len(accs)
        all_text = " ".join(str(v) for v in group["Clinical information data (Pseudo reports)"].values).lower()
        scenario = "B (RECIST)" if any(kw in all_text for kw in ["neoplasia", "clinical trial", "oncolog", "carcinoma", "metast"]) else "A (Lung-RADS)"

        added = False
        for acc in accs:
            has_dicom = (PATIENTS_DIR / acc / "original").exists() and any((PATIENTS_DIR / acc / "original").glob("*.dcm"))
            has_report = (PATIENTS_DIR / acc / "rapport_radiologique.md").exists()
            if has_dicom or has_report:
                tag = " ✅ Rapport prêt" if has_report else " 📂 DICOM local"
                choices.append(f"{pid} | Acc {acc} | {n} exams | {scenario} |{tag}")
                added = True

        if not added and orthanc_ok:
            # Only show patients needing Orthanc if Orthanc is reachable
            choices.append(f"{pid} | Acc {accs[0]} | {n} exams | {scenario} | 🌐 Orthanc requis")

    if not choices:
        return ["Aucun patient disponible (pas de données locales, Orthanc inaccessible)"]

    return choices


def parse_selection(selection: str):
    """Extract patient_id and accession from dropdown selection."""
    if not selection or "|" not in selection:
        return None, None
    parts = selection.split("|")
    patient_id = parts[0].strip()
    acc_part = parts[1].strip()  # "Acc 10969511"
    accession = acc_part.replace("Acc ", "").strip()
    return patient_id, accession


# ── Load existing results ─────────────────────────────────────────
def load_existing_results(selection):
    """Load already-generated results without running the pipeline."""
    _, accession = parse_selection(selection)
    if not accession:
        return "Sélectionnez un patient.", "", "", None

    patient_dir = PATIENTS_DIR / accession

    # Report
    report_path = patient_dir / "rapport_radiologique.md"
    if not report_path.exists():
        return (
            f"Aucun résultat existant pour l'accession **{accession}**.\n\n"
            "Lancez le pipeline pour générer un rapport.",
            "", "", None,
        )
    report = report_path.read_text(encoding="utf-8")

    # Confidence card
    confidence_card = _build_confidence_card(patient_dir, accession)

    # Evolution chart
    chart = _build_evolution_chart(patient_dir)

    # Log
    log = f"Résultats chargés depuis le cache pour l'accession {accession}."

    return report, confidence_card, log, chart


def _build_confidence_card(patient_dir: Path, accession: str):
    """Build the confidence / verification markdown card."""
    lines = []

    # Patient summary
    summary_path = patient_dir / "patient_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ctx = summary.get("clinical_context", {})
        scenario = ctx.get("scenario", "?")
        scenario_name = ctx.get("scenario_name", "")
        criteria = ctx.get("criteria", "")
        evidence = ctx.get("evidence", [])

        lines.append("## Scénario clinique")
        icon = "🔬" if scenario == "B" else "🫁"
        lines.append(f"{icon} **Scénario {scenario}** — {scenario_name}")
        lines.append(f"Critères : **{criteria}**")
        if evidence:
            lines.append(f"Evidence : {', '.join(evidence)}")
        lines.append("")

        # Confidence scores
        findings = summary.get("findings", [])
        if findings and findings[0].get("confidence"):
            lines.append("## Scores de confiance")
            for i, f in enumerate(findings, 1):
                conf = f.get("confidence", {})
                score = conf.get("score", "?")
                mx = conf.get("max", 8)
                level = conf.get("level", "?")
                cls = f.get("recist_classification", f.get("lungrads_classification", ""))
                d = f.get("diameter_axial_max_mm", 0)
                icon = "🟢" if score >= 6 else ("🟡" if score >= 3 else "🔴")
                lines.append(f"{icon} **F{i}** — {d:.1f}mm — {cls} — **{level} ({score}/{mx})**")
            lines.append("")

    # Verification report
    verif_path = patient_dir / "verification_report.json"
    if verif_path.exists():
        verif = json.loads(verif_path.read_text(encoding="utf-8"))
        lines.append("## Vérification anti-hallucination")
        for check in verif.get("checks", []):
            status = check.get("status", "")
            item = check.get("item", "")
            if status == "OK":
                lines.append(f"✅ {item}")
            elif "NON VÉRIFIABLE" in status:
                lines.append(f"⚠️ {item} — {status}")
            else:
                lines.append(f"❌ {item} — {status}")
        lines.append(f"\n📊 **Score de fiabilité : {verif.get('reliability_score', '?')}**")
        rec = verif.get("recommendation", "")
        if rec:
            lines.append(f"💡 {rec}")
        lines.append("")

    # Self-correction
    audit_path = patient_dir / "audit_trail.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        sc = audit.get("self_correction", {})
        if sc.get("applied"):
            orig = sc.get("original_issues", 0)
            corr = sc.get("corrected_issues", 0)
            fixed = orig - corr
            lines.append(f"🔄 **Self-correction appliquée** : {orig} → {corr} problèmes ({fixed} corrigé(s))")
        elif sc.get("original_issues", 0) > 0:
            lines.append("⚠️ Self-correction tentée mais rejetée")
        lines.append("")

    # Quality
    quality_path = patient_dir / "quality_report.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        lines.append("## Qualité du rapport")
        score = quality.get("score", "?")
        lines.append(f"📋 **Score : {score}**")
        for check in quality.get("checks", []):
            icon = "✅" if check.get("status") == "OK" else "❌"
            lines.append(f"{icon} {check.get('item', '')}")
        lines.append("")

    return "\n".join(lines) if lines else "Aucune donnée de confiance disponible."


def _build_evolution_chart(patient_dir: Path):
    """Build a plotly chart showing lesion evolution."""
    summary_path = patient_dir / "patient_summary.json"
    if not summary_path.exists():
        return None

    # We need the tracking data. Try to reconstruct from Excel.
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    patient_id = summary.get("patient", {}).get("id", "")
    accession = summary.get("exam", {}).get("accession", "")

    if not EXCEL_PATH.exists() or not patient_id:
        return None

    df = pd.read_excel(str(EXCEL_PATH))
    patient_rows = df[df["PatientID"] == patient_id]
    if patient_rows.empty:
        return None

    # Parse lesion sizes per exam
    exams = []
    for _, row in patient_rows.iterrows():
        acc = str(row["AccessionNumber"])
        sizes_raw = str(row.get("lesion size in mm", ""))
        sizes = []
        for s in sizes_raw.strip().split("\n"):
            s = s.strip()
            if s:
                try:
                    sizes.append(float(s))
                except ValueError:
                    pass
        exams.append({"accession": acc, "sizes": sizes})

    # Sort: current exam last
    exams.sort(key=lambda e: (1 if e["accession"] == accession else 0, int(e["accession"])))

    if not exams:
        return None

    max_findings = max(len(e["sizes"]) for e in exams)
    exam_labels = [f"Exam {i+1}\n(acc {e['accession'][-4:]})" for i, e in enumerate(exams)]

    fig = go.Figure()
    colors = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899"]

    for fi in range(max_findings):
        y_vals = []
        for e in exams:
            if fi < len(e["sizes"]):
                y_vals.append(e["sizes"][fi])
            else:
                y_vals.append(None)
        fig.add_trace(go.Scatter(
            x=exam_labels,
            y=y_vals,
            mode="lines+markers",
            name=f"F{fi+1}",
            line=dict(color=colors[fi % len(colors)], width=3),
            marker=dict(size=10),
        ))

    fig.update_layout(
        title=dict(text=f"Évolution des lésions — Patient {patient_id}", font=dict(size=16)),
        xaxis_title="Examen",
        yaxis_title="Diamètre (mm)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Add 10mm threshold line for RECIST
    fig.add_hline(y=10, line_dash="dash", line_color="gray",
                  annotation_text="Seuil cible RECIST (10mm)",
                  annotation_position="bottom right")

    return fig


# ── Run pipeline with progress ────────────────────────────────────
def run_full_pipeline(selection, progress=gr.Progress()):
    """Run the full pipeline for a selected patient/accession."""
    _, accession = parse_selection(selection)
    if not accession:
        yield "Sélectionnez un patient.", "", "", None
        return

    gemini_key = load_gemini_key(None)
    if not gemini_key:
        yield (
            "**Erreur** : Clé Gemini non trouvée.\n\n"
            "Configurez `GEMINI_API_KEY` dans `./work/.env`.",
            "", "", None,
        )
        return

    log_lines = []

    def add_log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    patient_dir = PATIENTS_DIR / accession
    original_dir = patient_dir / "original"
    results_dir = PATIENTS_DIR / "_results"
    os.makedirs(str(original_dir), exist_ok=True)
    os.makedirs(str(results_dir), exist_ok=True)

    try:
        # Check if DICOM files already exist locally (skip Orthanc if so)
        existing_dcm = list(original_dir.glob("*.dcm")) if original_dir.exists() else []
        seg_result_dir = results_dir / accession
        existing_seg = list(seg_result_dir.glob("*.dcm")) if seg_result_dir.exists() else []

        if existing_dcm:
            # ── Skip steps 1-2: reuse local files ──
            progress(0.05, desc="📂 Fichiers DICOM locaux détectés...")
            n_slices = len(existing_dcm)
            log_text = add_log(f"📂 {n_slices} coupes DICOM déjà présentes — Orthanc non requis")
            yield "", "", log_text, None
        else:
            # ── Step 1: Discovery ──
            progress(0.05, desc="🔍 Connexion à Orthanc...")
            log_text = add_log("🔍 Connexion à Orthanc...")
            yield "", "", log_text, None

            try:
                session, base_url = get_orthanc_session()
            except ConnectionError:
                log_text = add_log(
                    "❌ Orthanc inaccessible et aucune donnée locale.\n"
                    "   Vérifiez votre connexion réseau ou utilisez 'Voir résultats existants'."
                )
                yield "**Erreur** : Orthanc inaccessible et pas de données locales pour cette accession.", "", log_text, None
                return

            study_id, series_id, patient_id = step_discovery(session, base_url, accession)
            log_text = add_log(f"✅ Étude trouvée — PatientID: {patient_id}")
            yield "", "", log_text, None

            # ── Step 2: Download ──
            progress(0.15, desc="📥 Téléchargement CT...")
            log_text = add_log("📥 Téléchargement des coupes CT...")
            yield "", "", log_text, None

            n_slices = step_download(session, base_url, series_id, str(original_dir))
            log_text = add_log(f"✅ {n_slices} coupes téléchargées")
            yield "", "", log_text, None

        # ── Step 3: Segmentation ──
        progress(0.25, desc="🧠 Segmentation AI...")
        log_text = add_log("🧠 Segmentation AI en cours...")
        yield "", "", log_text, None

        seg_path, info_text = step_segmentation(str(patient_dir), str(results_dir))
        log_text = add_log("✅ Segmentation terminée")
        yield "", "", log_text, None

        # ── Step 4: Analyse SEG ──
        progress(0.35, desc="📊 Analyse volumétrique...")
        log_text = add_log("📊 Analyse volumétrique des nodules...")
        yield "", "", log_text, None

        patient_info, exam_info, findings = step_analyse_seg(seg_path, str(original_dir))
        for f in findings:
            add_log(f"  • F{f['SegmentNumber']}: {f['diameter_axial_max_mm']:.1f}mm, {f['volume_mm3']:.0f}mm³")
        log_text = add_log(f"✅ {len(findings)} nodules analysés")
        yield "", "", log_text, None

        # ── Step 5: Clinical context ──
        progress(0.45, desc="📋 Intelligence clinique...")
        log_text = add_log("📋 Chargement du contexte clinique...")
        yield "", "", log_text, None

        current_clinical, history_text, clinical_ctx, excel_df = step_clinical_context(patient_info["id"], accession)
        scenario = clinical_ctx["scenario"]
        log_text = add_log(f"✅ Scénario {scenario} — {clinical_ctx['scenario_name']}")

        # 5b: classify
        findings = classify_findings(findings, clinical_ctx)
        for f in findings:
            cls = f.get("recist_classification", f.get("lungrads_classification", ""))
            add_log(f"  • {f['SegmentLabel']}: {f['diameter_axial_max_mm']:.1f}mm → {cls}")

        # 5c: tracking
        tracking_table = build_lesion_tracking_table(patient_info["id"], accession, findings, excel_df)
        log_text = add_log("✅ Tableau de suivi F1↔F1 construit")
        yield "", "", log_text, None

        # 5d: RECIST
        recist_eval = build_recist_evaluation(tracking_table, findings, clinical_ctx)
        if recist_eval:
            import re as _re
            m = _re.search(r'→ ÉVALUATION : (.+)', recist_eval)
            if m:
                log_text = add_log(f"✅ RECIST : {m.group(1)}")

        # 5e: motif
        motif = extract_exam_motif(current_clinical)

        # 5f: confidence
        progress(0.55, desc="🎯 Calcul des scores de confiance...")
        pixel_sp = 1.0
        try:
            ct_files_list = sorted([fn for fn in os.listdir(str(original_dir)) if fn.endswith(".dcm")])
            ct_ds = pydicom.dcmread(os.path.join(str(original_dir), ct_files_list[0]))
            pixel_sp = float(ct_ds.PixelSpacing[0])
        except Exception:
            pass
        has_excel = excel_df is not None
        for f in findings:
            conf = compute_confidence(f, tracking_table, pixel_sp, has_excel)
            f["confidence"] = conf
        conf_str = ", ".join(
            f"F{int(f['SegmentLabel'].replace('Finding.',''))}({f['confidence']['score']}/8)"
            for f in findings
        )
        log_text = add_log(f"✅ Confiance : {conf_str}")
        yield "", "", log_text, None

        # 5g: VDT
        progress(0.60, desc="📈 Calcul VDT...")
        vdt_results = compute_all_vdt(tracking_table, exam_info)
        vdt_text = format_vdt_text(vdt_results)

        # 5h: reasoning
        if clinical_ctx.get("scenario") == "B":
            reasoning_text = format_recist_reasoning(tracking_table, findings, clinical_ctx, recist_eval, vdt_results)
        else:
            reasoning_text = format_lungrads_reasoning(findings, tracking_table, vdt_results)

        # 5i: hounsfield
        hu_note = analyze_hounsfield(findings, exam_info)

        # 5j: audit
        audit = audit_data_completeness(patient_info, exam_info, findings, excel_df, patient_info["id"])
        audit["missing"] = [m for m in audit["missing"] if "Densité" not in m]
        audit["missing"].append(hu_note)
        log_text = add_log(f"✅ Audit : {len(audit['missing'])} données manquantes identifiées")
        yield "", "", log_text, None

        # 5k-5l: tagged data + sources trace
        tagged_data = format_tagged_data(
            patient_info, exam_info, findings, tracking_table,
            recist_eval, clinical_ctx, current_clinical,
        )
        sources_trace = build_sources_trace(
            patient_info, exam_info, findings, tracking_table,
            recist_eval, clinical_ctx, current_clinical, seg_path,
        )
        for label, v in vdt_results.items():
            if v.get("vdt_days") is not None:
                sources_trace["sources"].append({
                    "data": f"VDT {label} = {v['vdt_days']}j",
                    "source": "CALC", "method": "volume_doubling_time",
                })
        sources_path = patient_dir / "sources_trace.json"
        sources_path.write_text(json.dumps(sources_trace, indent=2, ensure_ascii=False), encoding="utf-8")

        # ── Step 6: Generate report ──
        progress(0.70, desc="🤖 Génération du rapport via Gemini...")
        log_text = add_log("🤖 Génération du rapport via Gemini...")
        yield "", "", log_text, None

        report = step_generate_report(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, gemini_key, clinical_ctx,
            tracking_table, recist_eval, audit, tagged_data,
            vdt_text, reasoning_text,
        )
        n_words = len(report.split())
        log_text = add_log(f"✅ Rapport généré ({n_words} mots)")
        yield "", "", log_text, None

        # ── Step 6b: Verification ──
        progress(0.80, desc="🔍 Vérification anti-hallucination...")
        log_text = add_log("🔍 Vérification anti-hallucination...")
        yield "", "", log_text, None

        verification = verify_report(report, findings, tracking_table, clinical_ctx, recist_eval)
        verif_path = patient_dir / "verification_report.json"
        verif_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

        issues = [c for c in verification["checks"] if c["status"] != "OK"]
        log_text = add_log(f"📊 Fiabilité initiale : {verification['reliability_score']} ({len(issues)} problème(s))")

        # ── Step 6b2: Self-correction ──
        self_correction_info = {"applied": False, "original_issues": len(issues), "corrected_issues": 0}

        if issues:
            progress(0.85, desc="🔄 Self-correction en cours...")
            log_text = add_log("🔄 Tentative de self-correction...")
            yield "", "", log_text, None

            correction_prompt = build_correction_prompt(report, issues, findings)
            if correction_prompt:
                corrected_report, corr_model = call_gemini_correction(correction_prompt, gemini_key)
                if corrected_report:
                    corrected_verification = verify_report(
                        corrected_report, findings, tracking_table, clinical_ctx, recist_eval
                    )
                    corrected_issues = [c for c in corrected_verification["checks"] if c["status"] != "OK"]
                    self_correction_info["corrected_issues"] = len(corrected_issues)

                    if len(corrected_issues) <= len(issues):
                        report = corrected_report
                        verification = corrected_verification
                        self_correction_info["applied"] = True
                        verif_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")
                        fixed = len(issues) - len(corrected_issues)
                        log_text = add_log(f"✅ Self-correction acceptée : {len(issues)} → {len(corrected_issues)} problèmes ({fixed} corrigé(s))")
                    else:
                        log_text = add_log(f"⚠️ Self-correction rejetée ({len(issues)} → {len(corrected_issues)})")
        else:
            log_text = add_log("✅ Aucune hallucination détectée")
        yield "", "", log_text, None

        # Save audit trail
        audit_trail = {"self_correction": self_correction_info}
        (patient_dir / "audit_trail.json").write_text(
            json.dumps(audit_trail, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Step 6c: Quality ──
        progress(0.90, desc="📋 Auto-évaluation qualité...")
        quality = evaluate_report_quality(report, findings, clinical_ctx, verification)
        (patient_dir / "quality_report.json").write_text(
            json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log_text = add_log(f"✅ Qualité : {quality['score']}")

        # ── Step 7: Save ──
        progress(0.95, desc="💾 Sauvegarde...")
        step_output(str(patient_dir), patient_info, exam_info, findings, info_text, report, clinical_ctx)
        log_text = add_log("💾 Fichiers sauvegardés")
        log_text = add_log(f"\n🎉 Pipeline terminé avec succès !")

        progress(1.0, desc="✅ Terminé !")

        # Build outputs
        confidence_card = _build_confidence_card(patient_dir, accession)
        chart = _build_evolution_chart(patient_dir)

        yield report, confidence_card, log_text, chart

    except Exception as e:
        error_msg = f"❌ **Erreur** : {e}\n\n```\n{traceback.format_exc()}\n```"
        log_text = add_log(f"\n❌ ERREUR : {e}")
        yield error_msg, "", log_text, None


# ── History tab ───────────────────────────────────────────────────
def load_patient_history(selection):
    """Load all exam history for a patient."""
    patient_id, _ = parse_selection(selection)
    if not patient_id or not EXCEL_PATH.exists():
        return "Sélectionnez un patient.", None

    df = pd.read_excel(str(EXCEL_PATH))
    patient_rows = df[df["PatientID"] == patient_id].copy()
    if patient_rows.empty:
        return f"Aucune donnée pour {patient_id}.", None

    # Clinical context
    clinical_ctx = classify_clinical_context(patient_id, df)

    lines = [
        f"## Patient {patient_id}",
        f"**Scénario** : {clinical_ctx['scenario']} — {clinical_ctx['scenario_name']}",
        f"**Critères** : {clinical_ctx['criteria']}",
        f"**Evidence** : {', '.join(clinical_ctx['evidence'])}",
        f"**Nombre d'examens** : {len(patient_rows)}",
        "",
    ]

    # Table of exams
    table_data = []
    for _, row in patient_rows.iterrows():
        acc = str(row["AccessionNumber"])
        sizes_raw = str(row.get("lesion size in mm", ""))
        sizes = [s.strip() for s in sizes_raw.strip().split("\n") if s.strip()]
        clinical = str(row.get("Clinical information data (Pseudo reports)", ""))[:100]
        has_report = (PATIENTS_DIR / acc / "rapport_radiologique.md").exists()
        status = "✅ Rapport généré" if has_report else "—"
        table_data.append({
            "Accession": acc,
            "Nb lésions": len(sizes),
            "Tailles (mm)": ", ".join(sizes[:6]),
            "Statut": status,
        })

    table_df = pd.DataFrame(table_data)

    # Evolution text
    lines.append("### Évolution des lésions")
    exams = []
    for _, row in patient_rows.iterrows():
        acc = str(row["AccessionNumber"])
        sizes_raw = str(row.get("lesion size in mm", ""))
        sizes = []
        for s in sizes_raw.strip().split("\n"):
            s = s.strip()
            if s:
                try:
                    sizes.append(float(s))
                except ValueError:
                    pass
        exams.append({"accession": acc, "sizes": sizes})

    exams.sort(key=lambda e: int(e["accession"]))
    max_findings = max((len(e["sizes"]) for e in exams), default=0)

    for fi in range(max_findings):
        parts = []
        for i, e in enumerate(exams):
            if fi < len(e["sizes"]):
                parts.append(f"{e['sizes'][fi]}mm")
            else:
                parts.append("—")
        lines.append(f"- **F{fi+1}** : {' → '.join(parts)}")

    return "\n".join(lines), table_df


# ── About tab content ─────────────────────────────────────────────
ABOUT_TEXT = """
## UNBOXED — Pipeline Agentique de Radiologie

**Hackathon GE Healthcare x Centrale Lyon — 2026**

### Le problème
Un radiologue analyse 50-100 dossiers/jour. En oncologie pulmonaire, comparer des nodules
sur 3-6 examens CT, calculer les variations RECIST, et rédiger un rapport conforme prend
15 à 30 minutes par patient.

### Notre solution
Un agent IA en **12 étapes** qui automatise la chaîne complète :

```
Orthanc PACS → Download CT → Segmentation AI → Analyse volumétrique
                                                      ↓
Excel clinique → Contexte + Historique → Intelligence clinique
                                                      ↓
                                        Construction du prompt
                                                      ↓
                                        LLM (Gemini) → Rapport
                                                      ↓
                                        Vérification anti-hallucination
                                                      ↓
                                        Self-correction automatique
                                                      ↓
                                        Auto-évaluation qualité (10/10)
                                                      ↓
                                        Validation radiologue
```

### Ce qui nous différencie

1. **Intelligence clinique** — L'agent comprend *pourquoi* il fait le rapport
   (scénario A: Lung-RADS vs scénario B: RECIST 1.1)
2. **Anti-hallucination prouvée** — Le système détecte et corrige automatiquement
   les inventions du LLM (localisations anatomiques, chiffres incorrects)
3. **VDT automatique** — Temps de Doublement Volumétrique calculé pour chaque lésion
4. **Raisonnement RECIST transparent** — Tous les calculs sont visibles et vérifiables
5. **Confidence scoring** — Le radiologue sait où concentrer sa relecture

### Stack technique

| Composant | Technologie |
|-----------|-------------|
| Pipeline | Python, pydicom, numpy, pandas |
| Segmentation | dcm_seg_nodules (GE Healthcare) |
| LLM | Google Gemini 2.5 Flash |
| Protocole agentique | MCP (Model Context Protocol) |
| PACS | Orthanc DICOM Server |
| Interface | Gradio |

### Éthique
- L'agent ne remplace **jamais** le radiologue
- Chaque rapport porte un **avertissement IA**
- Données **pseudonymisées** (pas de nom réel)
- **Audit trail** complet pour traçabilité médico-légale
"""


# ══════════════════════════════════════════════════════════════════
#  GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════════

patient_choices = load_patient_choices()

with gr.Blocks(title="UNBOXED — Agent Radiologique") as demo:

    gr.HTML(
        """
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="margin-bottom:5px;">🏥 UNBOXED — Agent Radiologique Intelligent</h1>
            <p style="color:#6b7280; font-size:1.15em; margin-top:0;">
                Hackathon GE Healthcare × Centrale Lyon 2026
            </p>
        </div>
        """
    )

    with gr.Tabs():
        # ── TAB 1: Generate report ──
        with gr.TabItem("🔬 Générer un rapport"):
            with gr.Row():
                patient_dropdown = gr.Dropdown(
                    choices=patient_choices,
                    label="Sélectionner un patient",
                    info="PatientID | AccessionNumber | Nombre d'examens | Scénario (✅ = résultats existants)",
                    scale=3,
                )
                with gr.Column(scale=1, min_width=200):
                    run_btn = gr.Button("🚀 Lancer le pipeline", variant="primary", size="lg")
                    load_btn = gr.Button("📂 Voir résultats existants", variant="secondary", size="lg")

            progress_log = gr.Textbox(
                label="📋 Journal d'exécution",
                lines=12,
                max_lines=25,
                interactive=False,
            )

            with gr.Row():
                with gr.Column(scale=2):
                    report_output = gr.Markdown(
                        label="📄 Rapport radiologique",
                        value="*Sélectionnez un patient et lancez le pipeline.*",
                    )
                with gr.Column(scale=1):
                    confidence_output = gr.Markdown(
                        label="🎯 Fiche de confiance",
                        value="",
                    )

            evolution_chart = gr.Plot(label="📈 Évolution des lésions")

            # Wire buttons
            run_btn.click(
                fn=run_full_pipeline,
                inputs=[patient_dropdown],
                outputs=[report_output, confidence_output, progress_log, evolution_chart],
            )
            load_btn.click(
                fn=load_existing_results,
                inputs=[patient_dropdown],
                outputs=[report_output, confidence_output, progress_log, evolution_chart],
            )

        # ── TAB 2: History ──
        with gr.TabItem("📋 Historique patient"):
            history_dropdown = gr.Dropdown(
                choices=patient_choices,
                label="Sélectionner un patient",
            )
            history_btn = gr.Button("📊 Charger l'historique", variant="primary")

            history_text = gr.Markdown(value="*Sélectionnez un patient.*")
            history_table = gr.Dataframe(
                label="Examens disponibles",
                interactive=False,
            )

            history_btn.click(
                fn=load_patient_history,
                inputs=[history_dropdown],
                outputs=[history_text, history_table],
            )

        # ── TAB 3: About ──
        with gr.TabItem("ℹ️ À propos"):
            gr.Markdown(ABOUT_TEXT)


# ── Launch ────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
