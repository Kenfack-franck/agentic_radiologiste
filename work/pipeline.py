#!/usr/bin/env python3
"""
Pipeline complet agentique — Hackathon UNBOXED
Usage:
  python pipeline.py --accession 10969511 --gemini-key "AIza..."
  python pipeline.py --batch --gemini-key "AIza..."
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pydicom
import requests
from google import genai

from dcm_seg_nodules import extract_seg
from dcm_seg_nodules.registry import list_entries

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
EXCEL_PATH = os.path.join(BASE_DIR, "Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx")
PATIENTS_DIR = os.path.join(SCRIPT_DIR, "patients")

ORTHANC_URLS = [
    "http://10.0.1.215:8042",
    "https://orthanc.unboxed-2026.ovh",
]
ORTHANC_AUTH = ("unboxed", "unboxed2026")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-pro"]

# Keywords for clinical scenario classification
SCENARIO_B_KEYWORDS = [
    "neoplasia", "clinical trial", "psa", "recist",
    "m1", "metast", "progression", "non-target", "treatment",
    "chemotherapy", "oncolog", "carcinoma", "tumor", "tumour",
    "adenocarcinoma", "malignant", "stage iv", "stage iii",
]


# ── Helpers ──────────────────────────────────────────────────────────
def log(step, emoji, msg):
    print(f"\n{'='*60}")
    print(f"  {emoji}  ÉTAPE {step} — {msg}")
    print(f"{'='*60}")


def load_gemini_key(args_key):
    """Resolve Gemini API key from args, env, or .env file."""
    if args_key:
        return args_key
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def get_orthanc_session(orthanc_url=None):
    """Return (session, base_url) for a working Orthanc connection."""
    session = requests.Session()
    session.auth = ORTHANC_AUTH

    urls = [orthanc_url] if orthanc_url else ORTHANC_URLS
    for url in urls:
        try:
            r = session.get(f"{url}/system", timeout=5)
            if r.status_code == 200:
                print(f"  Orthanc connecté: {url}")
                return session, url
        except Exception:
            print(f"  {url} — timeout/erreur, essai suivant...")
    raise ConnectionError("Impossible de se connecter à Orthanc!")


# ── Clinical context classification ──────────────────────────────────
def classify_clinical_context(patient_id, excel_df=None):
    """Classify clinical scenario: A (nodule follow-up) or B (oncology/RECIST)."""
    if excel_df is None:
        if not os.path.exists(EXCEL_PATH):
            return {
                "scenario": "A",
                "scenario_name": "Suivi de nodule indéterminé",
                "criteria": "Lung-RADS / Fleischner",
                "clinical_indication": "Pas de contexte clinique disponible",
                "evidence": [],
            }
        excel_df = pd.read_excel(EXCEL_PATH)

    patient_rows = excel_df[excel_df["PatientID"] == patient_id]
    all_reports = " ".join(
        str(r.get("Clinical information data (Pseudo reports)", ""))
        for _, r in patient_rows.iterrows()
    ).lower()

    evidence = [kw for kw in SCENARIO_B_KEYWORDS if kw in all_reports]

    if evidence:
        return {
            "scenario": "B",
            "scenario_name": "Suivi oncologique / réponse au traitement",
            "criteria": "RECIST 1.1",
            "clinical_indication": "Cancer pulmonaire connu — évaluation de la réponse thérapeutique",
            "evidence": evidence,
        }
    else:
        return {
            "scenario": "A",
            "scenario_name": "Suivi de nodule indéterminé",
            "criteria": "Lung-RADS / Fleischner",
            "clinical_indication": "Surveillance de nodule(s) pulmonaire(s) — caractérisation",
            "evidence": ["Aucun mot-clé oncologique détecté"],
        }


# ── TÂCHE 1.1 — Extraction du motif d'examen ──────────────────────
def extract_exam_motif(current_clinical):
    """Extract the exam indication from the CLINICAL INFORMATION section."""
    if not current_clinical or current_clinical.strip() == "" or current_clinical == "nan":
        return "Non précisée — à compléter par le clinicien prescripteur"
    text = current_clinical.strip()
    # Try to extract text after "CLINICAL INFORMATION"
    match = re.search(r"CLINICAL\s+INFORMATION[.\s]*(.*?)(?:\n[A-Z]{2,}|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        motif = match.group(1).strip().rstrip(".")
        if motif:
            return motif
    # Fallback: first sentence/line
    first = text.split("\n")[0].strip()
    if first:
        return first
    return "Non précisée — à compléter par le clinicien prescripteur"


# ── TÂCHE 1.2 — Classification lésion cible/non-cible ─────────────
def classify_findings(findings, clinical_ctx):
    """Add recist_classification or lungrads_classification to each finding."""
    scenario = clinical_ctx.get("scenario", "A")

    if scenario == "B":
        # RECIST 1.1: sort by size descending, max 2 targets for lung
        sorted_idx = sorted(range(len(findings)),
                            key=lambda i: findings[i]["diameter_axial_max_mm"],
                            reverse=True)
        target_count = 0
        for idx in sorted_idx:
            d = findings[idx]["diameter_axial_max_mm"]
            if d >= 10.0 and target_count < 2:
                findings[idx]["recist_classification"] = "LÉSION CIBLE"
                target_count += 1
            elif d >= 6.0:
                findings[idx]["recist_classification"] = "LÉSION NON-CIBLE"
            else:
                findings[idx]["recist_classification"] = "NODULE INFRA-CENTIMÉTRIQUE"
    else:
        # Lung-RADS classification
        for f in findings:
            d = f["diameter_axial_max_mm"]
            if d >= 15.0:
                f["lungrads_classification"] = "Lung-RADS 4B (très suspect)"
            elif d >= 8.0:
                f["lungrads_classification"] = "Lung-RADS 4A (suspect)"
            elif d >= 6.0:
                f["lungrads_classification"] = "Lung-RADS 3 (probablement bénin)"
            else:
                f["lungrads_classification"] = "Lung-RADS 2 (bénin)"

    return findings


# ── TÂCHE 1.3 — Tableau de suivi lésionnel F1↔F1 ─────────────────
def build_lesion_tracking_table(patient_id, current_accession, current_findings, excel_df):
    """Build per-lesion tracking table across exams (F1↔F1 correspondence)."""
    if excel_df is None:
        return {"lesions": {}, "summary_table": "Aucun historique disponible."}

    patient_rows = excel_df[excel_df["PatientID"] == patient_id]
    if patient_rows.empty:
        return {"lesions": {}, "summary_table": "Aucun historique disponible."}

    # Collect all exams sorted by AccessionNumber (proxy chronological)
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

    # Sort: current exam always last, others by accession number ascending
    current_acc = str(current_accession)
    exams.sort(key=lambda e: (1 if e["accession"] == current_acc else 0, int(e["accession"])))

    # Determine max number of findings across all exams
    max_findings = max((len(e["sizes"]) for e in exams), default=0)
    # Also consider current SEG findings
    max_findings = max(max_findings, len(current_findings))

    # Build per-lesion history
    lesions = {}
    for fi in range(max_findings):
        label = f"F{fi + 1}"
        lesion_data = {"exams": []}
        for order, exam in enumerate(exams, 1):
            is_current = (exam["accession"] == str(current_accession))
            if fi < len(exam["sizes"]):
                lesion_data["exams"].append({
                    "accession": exam["accession"],
                    "size_mm": exam["sizes"][fi],
                    "order": order,
                    "current": is_current,
                })
            # If current exam but size from SEG differs, we'll use registry value
            # (the prompt already has SEG values separately)

        # Compute deltas
        sizes_list = [e["size_mm"] for e in lesion_data["exams"]]
        if len(sizes_list) >= 2:
            first_size = sizes_list[0]
            last_size = sizes_list[-1]
            prev_size = sizes_list[-2]
            delta_vs_first = ((last_size - first_size) / first_size * 100) if first_size > 0 else 0
            delta_vs_prev = ((last_size - prev_size) / prev_size * 100) if prev_size > 0 else 0
            lesion_data["delta_vs_first"] = f"{delta_vs_first:+.1f}%"
            lesion_data["delta_vs_previous"] = f"{delta_vs_prev:+.1f}%"
            if delta_vs_prev > 10:
                lesion_data["trend"] = "PROGRESSION"
            elif delta_vs_prev < -10:
                lesion_data["trend"] = "DIMINUTION"
            else:
                lesion_data["trend"] = "STABLE"
        elif len(sizes_list) == 1:
            lesion_data["delta_vs_first"] = "N/A"
            lesion_data["delta_vs_previous"] = "N/A"
            lesion_data["trend"] = "BASELINE"
        else:
            # New lesion not in previous exams
            lesion_data["trend"] = "NOUVEAU"

        lesions[label] = lesion_data

    # Build summary table text
    lines = ["TABLEAU D'ÉVOLUTION PAR LÉSION :\n"]
    for label, data in lesions.items():
        if not data["exams"]:
            lines.append(f"{label} : NOUVEAU — non présent dans les examens antérieurs")
            continue
        parts = []
        for i, e in enumerate(data["exams"]):
            tag = "ACTUEL" if e["current"] else f"exam {e['order']}"
            if i > 0:
                prev_size = data["exams"][i - 1]["size_mm"]
                pct = ((e["size_mm"] - prev_size) / prev_size * 100) if prev_size > 0 else 0
                parts.append(f"{e['size_mm']}mm ({tag}, {pct:+.0f}%)")
            else:
                parts.append(f"{e['size_mm']}mm ({tag})")
        trend = data.get("trend", "N/A")
        lines.append(f"{label} : {' → '.join(parts)} | {trend}")

    summary_table = "\n".join(lines)

    return {"lesions": lesions, "summary_table": summary_table}


def build_recist_evaluation(tracking_table, findings, clinical_ctx):
    """Compute RECIST 1.1 evaluation from tracking table (Scenario B only)."""
    if clinical_ctx.get("scenario") != "B":
        return ""

    # Identify target lesions (the ones classified as LÉSION CIBLE)
    target_labels = []
    for f in findings:
        if f.get("recist_classification") == "LÉSION CIBLE":
            seg_num = int(f["SegmentLabel"].replace("Finding.", ""))
            target_labels.append(f"F{seg_num}")

    if not target_labels:
        return "ÉVALUATION RECIST 1.1 : Aucune lésion cible identifiée (toutes < 10mm).\n"

    lesions = tracking_table.get("lesions", {})

    # Collect per-exam sums for target lesions only
    # First, find all exam accessions in order
    all_exams = {}  # order -> {accession, sum}
    for label in target_labels:
        if label not in lesions:
            continue
        for e in lesions[label]["exams"]:
            order = e["order"]
            if order not in all_exams:
                all_exams[order] = {"accession": e["accession"], "sum": 0, "current": e.get("current", False), "details": []}
            all_exams[order]["sum"] += e["size_mm"]
            all_exams[order]["details"].append(f"{label} ({e['size_mm']}mm)")

    if not all_exams:
        return "ÉVALUATION RECIST 1.1 : Données insuffisantes pour l'évaluation.\n"

    sorted_orders = sorted(all_exams.keys())
    baseline_order = sorted_orders[0]
    current_order = sorted_orders[-1]

    baseline_sum = all_exams[baseline_order]["sum"]
    current_sum = all_exams[current_order]["sum"]
    nadir_sum = min(all_exams[o]["sum"] for o in sorted_orders)
    nadir_order = min(sorted_orders, key=lambda o: all_exams[o]["sum"])

    delta_vs_baseline_pct = ((current_sum - baseline_sum) / baseline_sum * 100) if baseline_sum > 0 else 0
    delta_vs_nadir_pct = ((current_sum - nadir_sum) / nadir_sum * 100) if nadir_sum > 0 else 0
    delta_vs_nadir_abs = current_sum - nadir_sum

    # RECIST evaluation
    is_cr = current_sum == 0
    is_pr = delta_vs_baseline_pct <= -30
    is_pd = delta_vs_nadir_pct >= 20 and delta_vs_nadir_abs >= 5
    if is_cr:
        evaluation = "CR (Complete Response)"
    elif is_pd:
        evaluation = "PD (Progressive Disease)"
    elif is_pr:
        evaluation = "PR (Partial Response)"
    else:
        evaluation = "SD (Stable Disease)"

    lines = ["ÉVALUATION RECIST 1.1 :"]
    lines.append(f"  Lésions cibles : {' + '.join(all_exams[current_order]['details'])} = {current_sum:.1f}mm")
    lines.append(f"  Baseline (exam {baseline_order}, acc {all_exams[baseline_order]['accession']}) : "
                 f"{' + '.join(all_exams[baseline_order]['details'])} = {baseline_sum:.1f}mm")
    lines.append(f"  Nadir (exam {nadir_order}, acc {all_exams[nadir_order]['accession']}) : {nadir_sum:.1f}mm")
    lines.append(f"  Variation vs baseline : {delta_vs_baseline_pct:+.1f}%")
    lines.append(f"  Variation vs nadir : {delta_vs_nadir_pct:+.1f}% (absolue : {delta_vs_nadir_abs:+.1f}mm)")
    lines.append(f"  Seuil PR : ≤ -30% → {'OUI' if is_pr else 'NON'}")
    lines.append(f"  Seuil PD : ≥ +20% ET ≥ +5mm absolu → {'OUI' if is_pd else 'NON'}")
    lines.append(f"  → ÉVALUATION : {evaluation}")

    return "\n".join(lines) + "\n"


# ── TÂCHE 2.1 — Confidence score par finding ──────────────────────
def compute_confidence(finding, tracking_table, pixel_spacing, has_excel):
    """Compute confidence score (0-8) for a finding on 3 axes."""
    score = 0
    details = []

    # AXE 1 — Qualité de mesure (0-3)
    if finding.get("slices_with_nodule", 0) >= 3:
        score += 1
        details.append("+1 n_slices>=3")
    else:
        details.append("+0 n_slices<3")

    if finding.get("n_voxels", 0) >= 50:
        score += 1
        details.append("+1 n_voxels>=50")
    else:
        details.append("+0 n_voxels<50")

    if pixel_spacing <= 1.0:
        score += 1
        details.append("+1 pixel_spacing<=1mm")
    else:
        details.append("+0 pixel_spacing>1mm")

    # AXE 2 — Cohérence temporelle (0-3)
    seg_num = int(finding["SegmentLabel"].replace("Finding.", ""))
    label = f"F{seg_num}"
    lesion_history = tracking_table.get("lesions", {}).get(label, {})
    exams = lesion_history.get("exams", [])
    previous_exams = [e for e in exams if not e.get("current", False)]

    if len(previous_exams) >= 1:
        score += 1
        details.append("+1 exists_in_previous")
    else:
        details.append("+0 no_previous")

    if len(previous_exams) >= 1:
        current_exams = [e for e in exams if e.get("current", False)]
        if current_exams and previous_exams:
            prev_size = previous_exams[-1]["size_mm"]
            curr_size = current_exams[0]["size_mm"]
            variation = abs(curr_size - prev_size) / prev_size * 100 if prev_size > 0 else 999
            if variation <= 30:
                score += 1
                details.append(f"+1 variation<={variation:.0f}%")
            else:
                details.append(f"+0 variation={variation:.0f}%>30%")
        else:
            details.append("+0 no_current_match")

    if len(previous_exams) >= 2:
        score += 1
        details.append("+1 tracked>=2_exams")
    else:
        details.append("+0 tracked<2_exams")

    # AXE 3 — Complétude (0-2)
    if finding.get("volume_mm3", 0) > 0:
        score += 1
        details.append("+1 volume_available")
    else:
        details.append("+0 no_volume")

    if has_excel:
        score += 1
        details.append("+1 excel_available")
    else:
        details.append("+0 no_excel")

    # Level
    if score >= 6:
        level = "HAUTE"
    elif score >= 3:
        level = "MOYENNE"
    else:
        level = "BASSE"

    return {"score": score, "max": 8, "level": level, "details": details}


# ── TÂCHE 2.2 — Audit données manquantes ─────────────────────────
def audit_data_completeness(patient_info, exam_info, findings, excel_df, patient_id):
    """Check what data is available vs missing and generate warnings."""
    available = {
        "patient_id": patient_info.get("id", "N/A"),
        "patient_age": patient_info.get("age", "N/A"),
        "patient_sex": patient_info.get("sex", "N/A"),
        "modality": exam_info.get("modality", "N/A"),
        "slice_thickness": f"{exam_info.get('slice_thickness', 0)}mm",
        "nb_findings": len(findings),
    }

    if excel_df is not None:
        n_hist = len(excel_df[excel_df["PatientID"] == patient_id]) - 1
        available["historical_exams"] = max(n_hist, 0)
    else:
        available["historical_exams"] = 0

    missing = [
        "Injection de contraste : non spécifié dans les métadonnées DICOM",
        "Localisation anatomique des nodules : non disponible (segmentation sans mapping anatomique lobaire)",
        "Densité des nodules (solide/verre dépoli) : non évaluée dans cette version",
        "Latéralité des lésions : non précisée par l'algorithme de segmentation",
    ]

    warnings = []
    st = exam_info.get("slice_thickness", 0)
    if st >= 5.0:
        warnings.append(f"Épaisseur de coupe {st}mm : résolution limitée pour les nodules < 6mm")
    if available["historical_exams"] == 0:
        warnings.append("Aucun examen antérieur disponible pour comparaison")

    return {"available": available, "missing": missing, "warnings": warnings}


# ── TÂCHE 2.3 — Traçabilité source par donnée ────────────────────
def build_sources_trace(patient_info, exam_info, findings, tracking_table,
                        recist_eval, clinical_ctx, current_clinical, seg_path):
    """Build structured source traceability for every data point."""
    timestamp = datetime.datetime.now().isoformat()
    sources = []

    # DICOM sources
    for key in ["id", "age", "sex"]:
        sources.append({
            "data": f"Patient {key} = {patient_info.get(key, 'N/A')}",
            "source": "DICOM", "tag": f"Patient{key.title()}",
        })
    for key in ["modality", "slice_thickness", "manufacturer", "date"]:
        sources.append({
            "data": f"{key} = {exam_info.get(key, 'N/A')}",
            "source": "DICOM", "tag": key,
        })

    # SEG sources
    for i, f in enumerate(findings, 1):
        sources.append({
            "data": f"F{i} diameter = {f['diameter_axial_max_mm']}mm",
            "source": "SEG", "file": os.path.basename(seg_path),
            "segment": i,
            "method": "max_axial_cross_section",
            "n_voxels": f.get("n_voxels", 0),
            "n_slices": f.get("slices_with_nodule", 0),
        })
        sources.append({
            "data": f"F{i} volume = {f['volume_mm3']}mm3",
            "source": "SEG", "file": os.path.basename(seg_path),
            "segment": i,
            "method": "voxel_count_x_spacing",
        })

    # EXCEL sources
    lesions = tracking_table.get("lesions", {})
    for label, data in lesions.items():
        for e in data.get("exams", []):
            if not e.get("current", False):
                sources.append({
                    "data": f"Antérieur {label} = {e['size_mm']}mm",
                    "source": "EXCEL",
                    "column": "lesion size in mm",
                    "accession": e["accession"],
                })

    # CALC sources
    if recist_eval:
        sources.append({
            "data": f"RECIST evaluation",
            "source": "CALC",
            "method": "RECIST 1.1 criteria",
            "detail": recist_eval.strip(),
        })

    # ALGO sources
    sources.append({
        "data": f"Scénario = {clinical_ctx.get('scenario', 'N/A')}",
        "source": "ALGO",
        "method": "keyword_classification",
        "evidence": clinical_ctx.get("evidence", []),
    })

    # Motif
    motif = extract_exam_motif(current_clinical)
    sources.append({
        "data": f"Motif = {motif}",
        "source": "EXCEL",
        "column": "Clinical information data (Pseudo reports)",
    })

    return {"generation_timestamp": timestamp, "sources": sources}


def format_tagged_data(patient_info, exam_info, findings, tracking_table,
                       recist_eval, clinical_ctx, current_clinical):
    """Format data with [SRC:...] tags for prompt injection."""
    lines = ["=== DONNÉES TRACÉES (chaque donnée est taggée avec sa source) ==="]

    lines.append(f"[SRC:DICOM] PatientID = {patient_info['id']}")
    lines.append(f"[SRC:DICOM] PatientAge = {patient_info['age']}, PatientSex = {patient_info['sex']}")
    lines.append(f"[SRC:DICOM] Modality = {exam_info['modality']}, SliceThickness = {exam_info['slice_thickness']}mm")
    lines.append(f"[SRC:DICOM] Manufacturer = {exam_info['manufacturer']}")
    lines.append(f"[SRC:DICOM] StudyDate = {exam_info['date']}, AccessionNumber = {exam_info['accession']}")

    for i, f in enumerate(findings, 1):
        cls = f.get("recist_classification", f.get("lungrads_classification", ""))
        conf = f.get("confidence", {})
        conf_str = f"Confiance: {conf.get('level', '?')} ({conf.get('score', '?')}/{conf.get('max', 8)})" if conf else ""
        lines.append(f"[SRC:SEG] F{i} diamètre = {f['diameter_axial_max_mm']:.2f}mm "
                     f"(calculé sur {f['slices_with_nodule']} coupes, {f['n_voxels']} voxels) "
                     f"— {cls} — {conf_str}")

    lesions = tracking_table.get("lesions", {}) if tracking_table else {}
    for label, data in lesions.items():
        for e in data.get("exams", []):
            if not e.get("current", False):
                lines.append(f"[SRC:EXCEL] Antérieur exam {e['accession']} : {label}={e['size_mm']}mm")

    motif = extract_exam_motif(current_clinical)
    lines.append(f'[SRC:EXCEL] Motif : "{motif}"')

    for label, data in lesions.items():
        if "delta_vs_previous" in data:
            lines.append(f"[SRC:CALC] Variation {label} vs précédent : {data['delta_vs_previous']}")

    lines.append(f"[SRC:ALGO] Scénario détecté : {clinical_ctx.get('scenario', '?')} "
                 f"({clinical_ctx.get('criteria', '?')}) — evidence: "
                 f"{', '.join(repr(e) for e in clinical_ctx.get('evidence', []))}")

    if recist_eval:
        # Extract just the evaluation line
        for l in recist_eval.split("\n"):
            if "ÉVALUATION :" in l:
                lines.append(f"[SRC:CALC] {l.strip()}")

    return "\n".join(lines)


# ── TÂCHE 2.4 — Vérification post-génération ─────────────────────
def verify_report(report_text, findings, tracking_table, clinical_ctx, recist_eval):
    """Post-generation verification: check report against source data."""
    checks = []

    # 1. Count nodules mentioned
    nodule_patterns = re.findall(r'(?:F([1-9])\b|[Nn]odule\s*([1-9])\b|[Ll]ésion\s*(?:[Cc]ible\s*)?([1-9])\b|Finding[\s.]*([1-9])\b)', report_text)
    unique_refs = set()
    for m in nodule_patterns:
        for g in m:
            if g:
                unique_refs.add(int(g))
                break
    checks.append({
        "item": "Nombre nodules référencés",
        "expected": len(findings),
        "found_in_report": len(unique_refs),
        "status": "OK" if len(unique_refs) == len(findings) else
                  f"ATTENTION ({len(unique_refs)} vs {len(findings)} attendus)",
    })

    # 2. Check sizes mentioned — collect all mm values
    size_matches = re.findall(r'(\d+[.,]?\d*)\s*mm', report_text)
    report_sizes = []
    for s in size_matches:
        try:
            report_sizes.append(float(s.replace(",", ".")))
        except ValueError:
            pass

    # Known valid sizes: current findings + historical
    valid_sizes = set()
    for f in findings:
        valid_sizes.add(round(f["diameter_axial_max_mm"], 1))
        valid_sizes.add(round(f["diameter_sphere_mm"], 1))
        valid_sizes.add(round(f["volume_mm3"], 1))
    # Historical sizes
    for label, data in tracking_table.get("lesions", {}).items():
        for e in data.get("exams", []):
            valid_sizes.add(round(e["size_mm"], 1))
    # Slice thickness and other metadata
    valid_sizes.add(5.0)  # common slice thickness

    # RECIST computed sums (from recist_eval text)
    if recist_eval:
        sum_matches = re.findall(r'=\s*(\d+[.,]?\d*)\s*mm', recist_eval)
        for sm in sum_matches:
            try:
                valid_sizes.add(round(float(sm.replace(",", ".")), 1))
            except ValueError:
                pass
    # Also add intermediate sums across exams (e.g. sum of target lesions per exam)
    if tracking_table and tracking_table.get("lesions"):
        target_lesions = {k: v for k, v in tracking_table["lesions"].items()
                         if any(f.get("recist_classification") == "target" for f in findings
                                if f.get("label", "").lower() == k.lower())}
        if target_lesions:
            # Compute sum per exam index
            n_exams = max(len(v.get("exams", [])) for v in target_lesions.values())
            for i in range(n_exams):
                exam_sum = 0
                for v in target_lesions.values():
                    exams = v.get("exams", [])
                    if i < len(exams):
                        exam_sum += exams[i]["size_mm"]
                if exam_sum > 0:
                    valid_sizes.add(round(exam_sum, 1))

    verified_sizes = 0
    unverified_sizes = []
    for rs in report_sizes:
        # Check with tolerance ±1.5mm
        if any(abs(rs - vs) <= 1.5 for vs in valid_sizes):
            verified_sizes += 1
        elif rs < 1 or rs > 500:
            # Skip percentages/irrelevant numbers
            pass
        else:
            unverified_sizes.append(rs)

    total_size_checks = verified_sizes + len(unverified_sizes)
    checks.append({
        "item": "Tailles vérifiées",
        "expected": "toutes les tailles correspondent aux sources",
        "verified": verified_sizes,
        "unverified": unverified_sizes[:5],  # limit display
        "status": "OK" if not unverified_sizes else
                  f"ATTENTION ({len(unverified_sizes)} tailles non trouvées dans les sources)",
    })

    # 3. Check RECIST/Lung-RADS classification
    if clinical_ctx.get("scenario") == "B" and recist_eval:
        recist_match = re.search(r'→ ÉVALUATION : (\w+)', recist_eval)
        expected_cls = recist_match.group(1) if recist_match else "?"
        report_cls = None
        # Priority 1: Look for conclusion patterns (most reliable)
        conclusion_patterns = [
            r'(?:ÉVALUATION|évaluation)\s+RECIST[^:]*:\s*(?:\*\*)?\s*(Progression de la Maladie|Maladie Stable|Réponse Partielle|Réponse Complète|Progressive Disease|Stable Disease|Partial Response|Complete Response)\s*\(?([A-Z]{2})\)?',
            r'(?:ÉVALUATION|évaluation)\s+RECIST[^:]*:\s*(?:\*\*)?\s*(PD|SD|PR|CR)\b',
            r'(?:→|révèle une?|indique une?|conclut? à une?)\s+(?:\*\*)?\s*(Progression de la Maladie|Maladie Stable|Réponse Partielle|Réponse Complète)\s*\(?(PD|SD|PR|CR)\)?',
            r'(?:→|révèle une?|indique une?|conclut? à une?)\s+(?:\*\*)?\s*(PD|SD|PR|CR)\b',
        ]
        for pattern in conclusion_patterns:
            m = re.search(pattern, report_text, re.IGNORECASE)
            if m:
                found = m.group(m.lastindex).upper()
                mapping = {"MALADIE STABLE": "SD", "RÉPONSE PARTIELLE": "PR",
                           "RÉPONSE COMPLÈTE": "CR", "PROGRESSION DE LA MALADIE": "PD",
                           "STABLE DISEASE": "SD", "PARTIAL RESPONSE": "PR",
                           "COMPLETE RESPONSE": "CR", "PROGRESSIVE DISEASE": "PD"}
                report_cls = mapping.get(found, found)
                break
        checks.append({
            "item": "Classification RECIST",
            "expected": expected_cls,
            "found_in_report": report_cls or "NON TROUVÉ",
            "status": "OK" if report_cls == expected_cls else
                      f"DIVERGENCE (attendu {expected_cls}, trouvé {report_cls})",
        })

    # 4. Detect unverifiable claims
    unverifiable = []
    loc_patterns = re.findall(
        r'(?:lobe\s+(?:supérieur|inférieur|moyen)|lingula|apex|hile|para-hilaire|'
        r'sous[- ]pleur|segment\s+\w+|LSD|LSG|LID|LIG|LM)',
        report_text, re.IGNORECASE
    )
    if loc_patterns:
        unverifiable.append({
            "claim": f"Localisation anatomique mentionnée: {', '.join(set(loc_patterns[:3]))}",
            "source": "AUCUNE",
            "status": "NON VÉRIFIABLE (pas dans les sources)",
        })

    density_patterns = re.findall(
        r'(?:verre dépoli|ground[- ]glass|solide|sub[- ]?solide|mixte|calcifi|spicul|irréguli)',
        report_text, re.IGNORECASE
    )
    if density_patterns:
        unverifiable.append({
            "claim": f"Morphologie/densité mentionnée: {', '.join(set(density_patterns[:3]))}",
            "source": "AUCUNE",
            "status": "NON VÉRIFIABLE (pas évaluée par la segmentation)",
        })

    for uv in unverifiable:
        checks.append({"item": uv["claim"], "source": uv["source"], "status": uv["status"]})

    # Summary
    ok_count = sum(1 for c in checks if c["status"] == "OK")
    unverifiable_count = sum(1 for c in checks if "NON VÉRIFIABLE" in c.get("status", ""))
    incorrect_count = sum(1 for c in checks if "DIVERGENCE" in c.get("status", ""))
    attention_count = sum(1 for c in checks if "ATTENTION" in c.get("status", ""))
    total = len(checks)

    recommendation_items = []
    if incorrect_count:
        recommendation_items.append("classification RECIST")
    if unverifiable_count:
        recommendation_items.append("localisations/morphologies non vérifiables")
    if attention_count:
        recommendation_items.append("tailles non confirmées")

    return {
        "checks": checks,
        "verified_count": ok_count,
        "unverifiable_count": unverifiable_count,
        "incorrect_count": incorrect_count,
        "attention_count": attention_count,
        "reliability_score": f"{ok_count}/{total}",
        "recommendation": f"Relecture recommandée sur : {', '.join(recommendation_items)}" if recommendation_items
                          else "Toutes les données vérifiées — confiance élevée",
    }


# ── TÂCHE 3.1 — Temps de doublement volumétrique (VDT) ────────────
def compute_vdt(d1_mm, d2_mm, days_between):
    """Compute Volume Doubling Time from two diameters and interval in days."""
    if d1_mm <= 0 or d2_mm <= 0:
        return {"vdt_days": None, "classification": "Non calculable (diamètre nul)", "source_days": None}
    v1 = (4.0 / 3.0) * math.pi * (d1_mm / 2.0) ** 3
    v2 = (4.0 / 3.0) * math.pi * (d2_mm / 2.0) ** 3
    if abs(v2 - v1) < 0.001:
        return {"vdt_days": float("inf"), "classification": "Stable (pas de variation mesurable)", "source_days": days_between}
    vdt = (days_between * math.log(2)) / math.log(v2 / v1)
    if vdt < 0:
        cls = "Régression volumétrique"
    elif vdt < 200:
        cls = "Croissance rapide — hautement suspect de malignité"
    elif vdt < 400:
        cls = "Croissance modérée — suspect"
    elif vdt < 600:
        cls = "Croissance lente — indéterminé"
    else:
        cls = "Croissance très lente — probablement bénin"
    return {"vdt_days": round(vdt, 0), "classification": cls, "source_days": days_between}


def compute_all_vdt(tracking_table, exam_info):
    """Compute VDT for all lesions with at least 2 exam points."""
    study_date = exam_info.get("date", "")
    vdt_results = {}
    for label, data in tracking_table.get("lesions", {}).items():
        exams = data.get("exams", [])
        if len(exams) < 2:
            vdt_results[label] = {"vdt_days": None, "classification": "Non calculable (examen unique)",
                                  "d1": None, "d2": None, "days": None, "date_source": "N/A"}
            continue
        prev = exams[-2]
        curr = exams[-1]
        # Try to estimate days between exams — default 90 days
        days = 90
        date_source = "estimation (~90j entre examens)"
        if study_date and len(study_date) == 8:
            try:
                current_date = datetime.datetime.strptime(study_date, "%Y%m%d")
                # We don't have previous exam dates from Excel, use 90-day estimate
                days = 90
                date_source = "estimation (~90j, dates exactes non disponibles dans l'Excel)"
            except ValueError:
                pass
        vdt = compute_vdt(prev["size_mm"], curr["size_mm"], days)
        vdt_results[label] = {
            **vdt,
            "d1": prev["size_mm"], "d2": curr["size_mm"],
            "days": days, "date_source": date_source,
        }
    return vdt_results


def format_vdt_text(vdt_results):
    """Format VDT results for prompt injection."""
    lines = ["TEMPS DE DOUBLEMENT VOLUMÉTRIQUE (VDT) :"]
    for label, v in vdt_results.items():
        if v.get("vdt_days") is None:
            lines.append(f"  {label} : Non calculable (examen unique)")
        elif v["vdt_days"] == float("inf"):
            lines.append(f"  {label} : Stable (VDT = ∞) [{v['d1']}mm → {v['d2']}mm sur ~{v['days']}j {v['date_source']}]")
        else:
            vdt_str = f"{abs(v['vdt_days']):.0f}" if v["vdt_days"] != float("inf") else "∞"
            lines.append(f"  {label} : VDT = {vdt_str}j ({v['classification']}) "
                         f"[{v['d1']}mm → {v['d2']}mm sur ~{v['days']}j {v['date_source']}]")
    return "\n".join(lines)


# ── TÂCHE 3.2 — Raisonnement RECIST/Lung-RADS transparent ────────
def format_recist_reasoning(tracking_table, findings, clinical_ctx, recist_eval, vdt_results=None):
    """Build detailed RECIST reasoning block for prompt injection (Scenario B)."""
    if clinical_ctx.get("scenario") != "B":
        return ""
    target_findings = [f for f in findings if f.get("recist_classification") == "LÉSION CIBLE"]
    nontarget_findings = [f for f in findings if f.get("recist_classification") != "LÉSION CIBLE"]
    lesions = tracking_table.get("lesions", {})

    lines = ["═══ ÉVALUATION RECIST 1.1 — Raisonnement détaillé ═══", ""]
    lines.append("LÉSIONS CIBLES (≥10mm, max 2 poumon) :")
    for f in target_findings:
        sn = int(f["SegmentLabel"].replace("Finding.", ""))
        lines.append(f"  • F{sn} : {f['diameter_axial_max_mm']:.1f}mm (cible) — "
                     f"Confiance: {f.get('confidence', {}).get('level', '?')} ({f.get('confidence', {}).get('score', '?')}/8)")

    lines.append("\nLÉSIONS NON-CIBLES :")
    for f in nontarget_findings:
        sn = int(f["SegmentLabel"].replace("Finding.", ""))
        cls = f.get("recist_classification", "")
        lines.append(f"  • F{sn} : {f['diameter_axial_max_mm']:.1f}mm ({cls.lower()})")

    # Sums per exam
    target_labels = [f"F{int(f['SegmentLabel'].replace('Finding.', ''))}" for f in target_findings]
    all_exams = {}
    for label in target_labels:
        if label not in lesions:
            continue
        for e in lesions[label]["exams"]:
            o = e["order"]
            if o not in all_exams:
                all_exams[o] = {"acc": e["accession"], "sum": 0, "details": [], "current": e.get("current", False)}
            all_exams[o]["sum"] += e["size_mm"]
            all_exams[o]["details"].append(f"{label}({e['size_mm']}mm)")

    if all_exams:
        sorted_o = sorted(all_exams.keys())
        lines.append("\nSOMME DES DIAMÈTRES CIBLES :")
        for o in sorted_o:
            e = all_exams[o]
            tag = "ACTUEL" if e["current"] else f"Exam {o} (acc {e['acc']})"
            lines.append(f"  • {tag} : {' + '.join(e['details'])} = {e['sum']:.1f}mm")

        baseline = all_exams[sorted_o[0]]["sum"]
        current = all_exams[sorted_o[-1]]["sum"]
        nadir = min(all_exams[o]["sum"] for o in sorted_o)
        nadir_o = min(sorted_o, key=lambda o: all_exams[o]["sum"])
        lines.append(f"  • Nadir : {nadir:.1f}mm (exam {nadir_o}, acc {all_exams[nadir_o]['acc']})")

        dvb = ((current - baseline) / baseline * 100) if baseline > 0 else 0
        dvn = ((current - nadir) / nadir * 100) if nadir > 0 else 0
        dabs = current - nadir

        lines.append(f"\nCALCULS :")
        lines.append(f"  • Variation vs baseline : ({current:.1f} - {baseline:.1f}) / {baseline:.1f} × 100 = {dvb:+.1f}%")
        lines.append(f"  • Variation vs nadir : ({current:.1f} - {nadir:.1f}) / {nadir:.1f} × 100 = {dvn:+.1f}%")
        lines.append(f"  • Augmentation absolue vs nadir : {current:.1f} - {nadir:.1f} = {dabs:.1f}mm")

        is_cr = current == 0
        is_pr = dvb <= -30
        is_pd = dvn >= 20 and dabs >= 5
        lines.append(f"\nAPPLICATION DES SEUILS :")
        lines.append(f"  • CR : toutes cibles = 0 → {'OUI' if is_cr else 'NON'}")
        lines.append(f"  • PR : variation vs baseline ≤ -30% → {'OUI' if is_pr else 'NON'} ({dvb:+.1f}%)")
        lines.append(f"  • PD : variation vs nadir ≥ +20% ET augmentation ≥ 5mm → "
                     f"{'OUI' if is_pd else 'NON'} ({dvn:+.1f}% / {dabs:.1f}mm)")
        evaluation = "CR" if is_cr else ("PD" if is_pd else ("PR" if is_pr else "SD"))
        lines.append(f"\n  → ÉVALUATION RECIST : {evaluation}")

    # Non-target summary
    lines.append(f"\nLÉSIONS NON-CIBLES :")
    for f in nontarget_findings:
        sn = int(f["SegmentLabel"].replace("Finding.", ""))
        label = f"F{sn}"
        trend = lesions.get(label, {}).get("trend", "N/A")
        delta = lesions.get(label, {}).get("delta_vs_previous", "N/A")
        lines.append(f"  {label} : {trend} ({delta})")
    lines.append("  Pas de progression non-équivoque des non-cibles.")
    lines.append("\nNOUVELLES LÉSIONS : Aucune détectée.")

    return "\n".join(lines)


def format_lungrads_reasoning(findings, tracking_table, vdt_results=None):
    """Build detailed Lung-RADS reasoning block for prompt injection (Scenario A)."""
    lesions = tracking_table.get("lesions", {})
    lines = ["═══ CLASSIFICATION LUNG-RADS — Raisonnement détaillé ═══", ""]

    highest_category = 0
    highest_label = ""

    for i, f in enumerate(findings, 1):
        label = f"F{i}"
        d = f["diameter_axial_max_mm"]
        lr = f.get("lungrads_classification", "N/A")
        trend = lesions.get(label, {}).get("trend", "N/A")
        delta = lesions.get(label, {}).get("delta_vs_previous", "N/A")

        if d >= 15:
            cat_num = 4
        elif d >= 8:
            cat_num = 3
        elif d >= 6:
            cat_num = 2
        else:
            cat_num = 1

        if cat_num > highest_category:
            highest_category = cat_num
            highest_label = label

        lines.append(f"• {label} : {d:.1f}mm → {lr}")
        if trend != "BASELINE" and trend != "NOUVEAU":
            lines.append(f"  Évolution : {trend} ({delta})")
        if vdt_results and label in vdt_results:
            v = vdt_results[label]
            if v.get("vdt_days") is not None and v["vdt_days"] != float("inf"):
                lines.append(f"  VDT : {abs(v['vdt_days']):.0f}j — {v['classification']}")

        # Fleischner recommendation
        if d >= 15:
            lines.append("  Recommandation Fleischner : PET-CT ou biopsie")
        elif d >= 8:
            lines.append("  Recommandation Fleischner : CT de suivi à 3 mois ou PET-CT")
        elif d >= 6:
            lines.append("  Recommandation Fleischner : CT de suivi à 6 mois")
        else:
            lines.append("  Recommandation Fleischner : CT de dépistage annuel")
        lines.append("")

    cat_map = {0: "1", 1: "2", 2: "3", 3: "4A", 4: "4B"}
    global_cat = cat_map.get(highest_category, "?")
    lines.append(f"→ CLASSIFICATION GLOBALE : Lung-RADS {global_cat} (basée sur {highest_label})")

    if highest_category >= 3:
        lines.append(f"→ RECOMMANDATION : CT de suivi à 3 mois ou PET-CT pour {highest_label}")
    elif highest_category >= 2:
        lines.append(f"→ RECOMMANDATION : CT de suivi à 6 mois")
    else:
        lines.append(f"→ RECOMMANDATION : CT de dépistage annuel")

    return "\n".join(lines)


# ── TÂCHE 3.3 — Analyse densité Hounsfield (Option B fallback) ───
def analyze_hounsfield(findings, exam_info):
    """Hounsfield density analysis — conservative fallback (Option B).
    Real CT/SEG alignment is too complex for hackathon; add structured note."""
    st = exam_info.get("slice_thickness", 5.0)
    hu_note = (f"Densité des nodules : non évaluée dans cette version. "
               f"Épaisseur de coupe de {st}mm limite la caractérisation précise. "
               f"Évaluation visuelle par le radiologue recommandée.")
    for f in findings:
        f["hounsfield_status"] = "Non évalué (alignement CT/SEG non implémenté)"
    return hu_note


# ── Self-correction — Boucle de correction post-vérification ─────
def build_correction_prompt(report_text, issues, findings):
    """Build a correction prompt from verify_report issues."""
    instructions = []
    for issue in issues:
        status = issue.get("status", "")
        item = issue.get("item", "")

        if "NON VÉRIFIABLE" in status:
            # Hallucinated location or morphology
            claim = issue.get("claim", item)
            instructions.append(
                f"- ERREUR D'HALLUCINATION : Tu as mentionné '{claim}', "
                f"mais cette information n'est PAS dans les données sources. "
                f"Retire cette affirmation et écris 'non précisé(e)' ou "
                f"'localisation non disponible (segmentation sans mapping anatomique)'."
            )
        elif "DIVERGENCE" in status:
            expected = issue.get("expected", "?")
            found = issue.get("found_in_report", "?")
            instructions.append(
                f"- ERREUR DE CLASSIFICATION : Tu as conclu '{found}' "
                f"mais le calcul correct donne '{expected}'. "
                f"Corrige la conclusion pour indiquer '{expected}'."
            )
        elif "ATTENTION" in status:
            if "tailles" in item.lower():
                unverified = issue.get("unverified", [])
                if unverified:
                    # Build valid sizes for reference
                    valid = []
                    for f in findings:
                        valid.append(f"{f['diameter_axial_max_mm']:.2f}mm")
                    instructions.append(
                        f"- ERREUR DE MESURE : Les tailles suivantes ne correspondent "
                        f"à aucune source : {unverified}. "
                        f"Les mesures réelles des nodules sont : {', '.join(valid)}. "
                        f"Vérifie et corrige les chiffres erronés."
                    )
            elif "nodules" in item.lower():
                expected = issue.get("expected", "?")
                found = issue.get("found_in_report", "?")
                instructions.append(
                    f"- ERREUR DE COMPTAGE : Tu as référencé {found} nodules "
                    f"mais il y en a {expected}. Corrige le rapport pour "
                    f"mentionner exactement {expected} nodules."
                )

    if not instructions:
        return None

    correction_prompt = f"""Voici un rapport radiologique que tu as généré :
---
{report_text}
---

Une vérification automatique a détecté les problèmes suivants :

{chr(10).join(instructions)}

CONSIGNES DE CORRECTION :
- Réécris le rapport en corrigeant UNIQUEMENT ces problèmes.
- Ne change RIEN d'autre : garde la même structure, le même style, les mêmes sections.
- Pour les localisations anatomiques non vérifiables : remplace par 'localisation non précisée par l'algorithme de segmentation'.
- Pour les densités/morphologies non vérifiables : remplace par 'densité non évaluée dans cette version'.
- Pour les chiffres incorrects : utilise la valeur correcte indiquée ci-dessus.
- Le rapport doit rester en français médical professionnel.
- Conserve tous les avertissements IA et sections de limitations.
"""
    return correction_prompt


def call_gemini_correction(correction_prompt, gemini_key):
    """Call Gemini with a correction prompt."""
    client = genai.Client(api_key=gemini_key)
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=correction_prompt,
                config={"temperature": 0.2, "max_output_tokens": 12000},
            )
            return response.text, model_name
        except Exception:
            continue
    return None, None


# ── TÂCHE 3.4 — Auto-évaluation qualité du rapport ───────────────
def evaluate_report_quality(report_text, findings, clinical_ctx, verification):
    """Evaluate report completeness with a 10-point checklist."""
    checks = []
    scenario = clinical_ctx.get("scenario", "A")

    # 1. Indication
    has_indication = bool(re.search(r'(?:indication|motif)', report_text, re.IGNORECASE))
    checks.append(("Indication clinique", has_indication))

    # 2. Technique
    has_technique = bool(re.search(r'(?:CT|scanner|technique|modalité|tomodensitom)', report_text, re.IGNORECASE))
    checks.append(("Technique d'examen", has_technique))

    # 3. All findings mentioned
    mentioned = set()
    for m in re.finditer(r'F(\d+)', report_text):
        mentioned.add(int(m.group(1)))
    n_expected = len(findings)
    n_found = len(mentioned.intersection(range(1, n_expected + 1)))
    checks.append((f"Findings ({n_found}/{n_expected})", n_found == n_expected))

    # 4. Sizes
    has_sizes = bool(re.search(r'\d+[.,]?\d*\s*mm', report_text))
    checks.append(("Taille actuelle", has_sizes))

    # 5. Comparison with prior exams
    has_comparison = bool(re.search(r'(?:antérieur|précédent|baseline|nadir|évolution)', report_text, re.IGNORECASE))
    checks.append(("Comparaison antérieurs", has_comparison))

    # 6. Classification
    if scenario == "B":
        has_cls = bool(re.search(r'(?:RECIST|CR|PR|SD|PD)', report_text, re.IGNORECASE))
    else:
        has_cls = bool(re.search(r'(?:Lung.?RADS|LR.?\d|Fleischner)', report_text, re.IGNORECASE))
    checks.append(("Classification fournie", has_cls))

    # 7. Visible reasoning
    has_reasoning = bool(re.search(r'(?:%|variation|somme|calcul|seuil)', report_text, re.IGNORECASE))
    checks.append(("Raisonnement visible", has_reasoning))

    # 8. Follow-up recommendation
    has_followup = bool(re.search(r'(?:contrôle|suivi|recommand|biopsie|PET|RCP)', report_text, re.IGNORECASE))
    checks.append(("Recommandation de suivi", has_followup))

    # 9. AI disclaimer
    has_ai = bool(re.search(r'(?:IA|intelligence artificielle|automatique|validé par un radiologue)', report_text, re.IGNORECASE))
    checks.append(("Avertissement IA", has_ai))

    # 10. Missing data mentioned
    has_missing = bool(re.search(r'(?:non disponible|non évaluab|limitation|manquant|non précisé)', report_text, re.IGNORECASE))
    checks.append(("Données manquantes signalées", has_missing))

    score = sum(1 for _, ok in checks if ok)
    total = len(checks)

    result = {
        "score": f"{score}/{total}",
        "checks": [],
    }
    for label, ok in checks:
        result["checks"].append({"item": label, "status": "OK" if ok else "MANQUANT"})

    return result


# ── Prompt Templates ────────────────────────────────────────────────
def build_prompt_template_a(patient_info, exam_info, findings, info_text,
                            current_clinical, history_text, clinical_ctx,
                            tracking_table=None, audit=None, tagged_data=None,
                            vdt_text=None, reasoning_text=None):
    """Template A: Nodule follow-up (Lung-RADS / Fleischner)."""
    findings_text = _format_findings(findings, clinical_ctx)
    motif = extract_exam_motif(current_clinical)
    tracking_text = tracking_table["summary_table"] if tracking_table else ""
    tagged_text = tagged_data if tagged_data else ""
    vdt_block = vdt_text if vdt_text else ""
    reasoning_block = reasoning_text if reasoning_text else ""
    missing_text = ""
    if audit:
        missing_text = "DONNÉES NON DISPONIBLES (ne pas inventer, mentionner comme 'non évaluable') :\n"
        for m in audit.get("missing", []):
            missing_text += f"- {m}\n"
        missing_text += "Le rapport DOIT mentionner ces limitations. NE PAS inventer de localisation ou de caractéristique morphologique."

    return f"""Tu es un radiologue expert en imagerie thoracique.

Le rapport DOIT commencer par : INDICATION : {motif}

CONTEXTE CLINIQUE : {clinical_ctx['scenario_name']}
CRITÈRES À APPLIQUER : {clinical_ctx['criteria']}

CONSIGNES STRICTES :
- La PREMIÈRE LIGNE du rapport doit être : "INDICATION : {motif}"
- Reproduis le raisonnement Lung-RADS détaillé ci-dessous dans le rapport
- Pour chaque nodule, mentionne le VDT si disponible
- NE PAS inventer de données non fournies (localisation, densité, etc.)
- Rédige en français médical professionnel

{tagged_text}

=== RÉSULTATS SEGMENTATION AI (MESURES ACTUELLES) ===
{findings_text}

=== ÉVOLUTION LÉSIONNELLE ===
{tracking_text if tracking_text else "Aucun historique disponible."}

=== {vdt_block}

=== {reasoning_block}

=== {missing_text}

=== FORMAT DU RAPPORT (sois concis, max 1500 mots) ===
1. INDICATION : {motif}
2. TECHNIQUE (modalité, paramètres, limitations)
3. RÉSULTATS — pour chaque nodule : identifiant, Lung-RADS, taille, VDT, confiance, évolution
4. CLASSIFICATION LUNG-RADS (reproduis le raisonnement ci-dessus)
5. CONCLUSION (Lung-RADS global, recommandation Fleischner, délai prochain contrôle)
6. LIMITATIONS (données non disponibles)
7. SOURCES DES DONNÉES
8. AVERTISSEMENT IA
"""


def build_prompt_template_b(patient_info, exam_info, findings, info_text,
                            current_clinical, history_text, clinical_ctx,
                            tracking_table=None, recist_eval=None,
                            audit=None, tagged_data=None,
                            vdt_text=None, reasoning_text=None):
    """Template B: Oncology follow-up (RECIST 1.1)."""
    findings_text = _format_findings(findings, clinical_ctx)
    motif = extract_exam_motif(current_clinical)
    tracking_text = tracking_table["summary_table"] if tracking_table else ""
    tagged_text = tagged_data if tagged_data else ""
    vdt_block = vdt_text if vdt_text else ""
    reasoning_block = reasoning_text if reasoning_text else ""
    missing_text = ""
    if audit:
        missing_text = "DONNÉES NON DISPONIBLES (ne pas inventer, mentionner comme 'non évaluable') :\n"
        for m in audit.get("missing", []):
            missing_text += f"- {m}\n"
        missing_text += "Le rapport DOIT mentionner ces limitations. NE PAS inventer de localisation ou de caractéristique morphologique."

    return f"""Tu es un radiologue expert en imagerie thoracique spécialisé en oncologie pulmonaire.

Le rapport DOIT commencer par : INDICATION : {motif}

CONTEXTE CLINIQUE : {clinical_ctx['scenario_name']}
CRITÈRES À APPLIQUER : {clinical_ctx['criteria']}
ÉLÉMENTS DÉCLENCHEURS : {', '.join(clinical_ctx['evidence'])}

CONSIGNES STRICTES :
- La PREMIÈRE LIGNE du rapport doit être : "INDICATION : {motif}"
- Reproduis le raisonnement RECIST détaillé ci-dessous dans une section dédiée
- Le radiologue doit voir TOUS les calculs, pas juste la conclusion
- Pour chaque lésion, mentionne le VDT si disponible
- NE PAS inventer de données non fournies (localisation, densité, etc.)
- Rédige en français médical professionnel, ton précis et quantitatif

{tagged_text}

=== RÉSULTATS SEGMENTATION AI (MESURES ACTUELLES) ===
{findings_text}

=== ÉVOLUTION LÉSIONNELLE ===
{tracking_text if tracking_text else "Aucun historique disponible."}

=== {vdt_block}

=== {reasoning_block}

=== {missing_text}

=== FORMAT DU RAPPORT (sois concis, max 1500 mots) ===
1. INDICATION : {motif}
2. TECHNIQUE (modalité, paramètres, limitations)
3. RÉSULTATS — pour chaque lésion : identifiant, classification, taille, VDT, confiance, évolution
4. ÉVALUATION RECIST 1.1 (reproduis le raisonnement détaillé ci-dessus avec calculs visibles)
5. CONCLUSION (évaluation globale, tendance, recommandation RCP)
6. LIMITATIONS (données non disponibles)
7. SOURCES DES DONNÉES
8. AVERTISSEMENT IA
"""


def _format_findings(findings, clinical_ctx=None):
    """Format findings with classification info."""
    text = f"Nombre de nodules détectés: {len(findings)}\n"
    for i, f in enumerate(findings, 1):
        label = f"F{i}"
        text += f"""
{label} / {f['SegmentLabel']}:
  - Diamètre axial maximal (MESURE ACTUELLE): {f['diameter_axial_max_mm']:.2f} mm
  - Diamètre sphère équivalente: {f['diameter_sphere_mm']:.2f} mm
  - Volume: {f['volume_mm3']:.2f} mm³ ({f['volume_cm3']:.4f} cm³)
  - Coupes contenant le nodule: {f['slices_with_nodule']}"""
        if "recist_classification" in f:
            text += f"\n  - Classification RECIST : {f['recist_classification']}"
        if "lungrads_classification" in f:
            text += f"\n  - Classification : {f['lungrads_classification']}"
        text += "\n"
    return text


# ── ÉTAPE 1: Discovery ──────────────────────────────────────────────
def step_discovery(session, base_url, accession):
    log(1, "🔍", f"DISCOVERY — Recherche AccessionNumber={accession}")

    studies = session.get(f"{base_url}/studies").json()
    print(f"  {len(studies)} études trouvées sur Orthanc")

    target_study_id = None
    patient_id = None
    for study_id in studies:
        info = session.get(f"{base_url}/studies/{study_id}").json()
        acc = info.get("MainDicomTags", {}).get("AccessionNumber", "")
        if acc == accession:
            target_study_id = study_id
            patient_id = info.get("PatientMainDicomTags", {}).get("PatientID", "")
            print(f"  Étude trouvée: {study_id}")
            print(f"  PatientID: {patient_id}")
            break

    if not target_study_id:
        raise ValueError(f"Étude avec AccessionNumber={accession} non trouvée!")

    # Find CT series — prefer "CEV torax", fallback to largest CT series
    study_data = session.get(f"{base_url}/studies/{target_study_id}").json()
    series_list = study_data.get("Series", [])

    best_ct_series = None
    best_ct_instances = 0
    target_series_id = None

    for series_id in series_list:
        s_info = session.get(f"{base_url}/series/{series_id}").json()
        tags = s_info.get("MainDicomTags", {})
        modality = tags.get("Modality", "")
        desc = tags.get("SeriesDescription", "")
        n_inst = len(s_info.get("Instances", []))
        print(f"    Série: {desc} | {modality} | {n_inst} instances")

        if modality == "CT":
            if "CEV" in desc.upper() and "TORAX" in desc.upper():
                target_series_id = series_id
                print(f"    >>> SÉRIE CIBLE (CEV torax)")
            elif n_inst > best_ct_instances:
                best_ct_series = series_id
                best_ct_instances = n_inst

    if not target_series_id:
        target_series_id = best_ct_series
        if target_series_id:
            print(f"    >>> Fallback: plus grande série CT ({best_ct_instances} instances)")
        else:
            raise ValueError("Aucune série CT trouvée!")

    return target_study_id, target_series_id, patient_id


# ── ÉTAPE 2: Download ───────────────────────────────────────────────
def step_download(session, base_url, series_id, output_dir):
    log(2, "⬇️", "DOWNLOAD — Téléchargement des instances DICOM")

    os.makedirs(output_dir, exist_ok=True)

    series_info = session.get(f"{base_url}/series/{series_id}").json()
    instances = series_info["Instances"]
    print(f"  {len(instances)} instances à télécharger")

    for i, inst_id in enumerate(instances):
        resp = session.get(f"{base_url}/instances/{inst_id}/file")
        resp.raise_for_status()
        filepath = os.path.join(output_dir, f"slice_{i:04d}.dcm")
        with open(filepath, "wb") as f:
            f.write(resp.content)
        if (i + 1) % 50 == 0 or i == 0 or i == len(instances) - 1:
            print(f"  Téléchargé {i+1}/{len(instances)}")

    print(f"  Terminé: {len(instances)} fichiers dans {output_dir}")
    return len(instances)


# ── ÉTAPE 3: Segmentation ───────────────────────────────────────────
def step_segmentation(patient_dir, results_dir):
    log(3, "🧠", "SEGMENTATION — Lancement de l'algorithme mock")

    seg_path, info_text = extract_seg(
        patient_dir,
        output_dir=results_dir,
        series_subdir="original",
    )
    print(f"  SEG généré: {seg_path}")
    print(f"  Info:\n{info_text}")
    return seg_path, info_text


# ── ÉTAPE 4: Analyse SEG ────────────────────────────────────────────
def step_analyse_seg(seg_path, original_dir):
    log(4, "📊", "ANALYSE SEG — Extraction des mesures nodulaires")

    seg_ds = pydicom.dcmread(seg_path)

    # Read a CT slice for fallback pixel info
    ct_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".dcm")])
    ct_ds = pydicom.dcmread(os.path.join(original_dir, ct_files[0]))

    # Patient & exam metadata
    patient_info = {
        "id": str(getattr(ct_ds, "PatientID", "N/A")),
        "age": str(getattr(ct_ds, "PatientAge", "N/A")),
        "sex": str(getattr(ct_ds, "PatientSex", "N/A")),
    }
    exam_info = {
        "date": str(getattr(ct_ds, "StudyDate", "N/A")),
        "description": str(getattr(ct_ds, "StudyDescription", "N/A")),
        "accession": str(getattr(ct_ds, "AccessionNumber", "N/A")),
        "modality": str(getattr(ct_ds, "Modality", "N/A")),
        "manufacturer": str(getattr(ct_ds, "Manufacturer", "N/A")),
        "slice_thickness": float(getattr(ct_ds, "SliceThickness", 0)),
    }

    # Pixel spacing from SEG or CT
    try:
        shared = seg_ds.SharedFunctionalGroupsSequence[0]
        pm = shared.PixelMeasuresSequence[0]
        pixel_spacing = [float(pm.PixelSpacing[0]), float(pm.PixelSpacing[1])]
        seg_slice_thickness = float(pm.SliceThickness)
    except Exception:
        pixel_spacing = [float(ct_ds.PixelSpacing[0]), float(ct_ds.PixelSpacing[1])]
        seg_slice_thickness = float(ct_ds.SliceThickness)

    print(f"  PixelSpacing: {pixel_spacing}, SliceThickness: {seg_slice_thickness}")

    # Segments
    segments = []
    for seg in seg_ds.SegmentSequence:
        s = {
            "SegmentNumber": int(seg.SegmentNumber),
            "SegmentLabel": str(getattr(seg, "SegmentLabel", "N/A")),
            "SegmentDescription": str(getattr(seg, "SegmentDescription", "N/A")),
            "SegmentAlgorithmType": str(getattr(seg, "SegmentAlgorithmType", "N/A")),
            "SegmentAlgorithmName": str(getattr(seg, "SegmentAlgorithmName", "N/A")),
        }
        try:
            s["CategoryCodeMeaning"] = str(seg.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning)
        except Exception:
            s["CategoryCodeMeaning"] = "N/A"
        try:
            s["TypeCodeMeaning"] = str(seg.SegmentedPropertyTypeCodeSequence[0].CodeMeaning)
        except Exception:
            s["TypeCodeMeaning"] = "N/A"
        segments.append(s)

    # Frame-to-segment mapping
    pixel_array = seg_ds.pixel_array
    frame_to_segment = {}
    if hasattr(seg_ds, "PerFrameFunctionalGroupsSequence"):
        for i, frame_fg in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
            try:
                ref = int(frame_fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                frame_to_segment[i] = ref
            except Exception:
                pass

    n_segments = len(segments)
    findings = []

    for seg_info in segments:
        seg_num = seg_info["SegmentNumber"]
        frame_indices = [i for i, sn in frame_to_segment.items() if sn == seg_num]

        if frame_indices:
            seg_frames = pixel_array[frame_indices]
        else:
            frames_per_seg = pixel_array.shape[0] // n_segments
            start = (seg_num - 1) * frames_per_seg
            seg_frames = pixel_array[start:start + frames_per_seg]

        n_voxels = int(np.count_nonzero(seg_frames))
        slices_with_nodule = int(np.sum(np.any(seg_frames > 0, axis=(1, 2))))

        voxel_vol = pixel_spacing[0] * pixel_spacing[1] * seg_slice_thickness
        volume_mm3 = n_voxels * voxel_vol
        volume_cm3 = volume_mm3 / 1000.0

        diameter_sphere = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0) if volume_mm3 > 0 else 0.0

        max_area_px = max((np.count_nonzero(frame) for frame in seg_frames), default=0)
        diameter_axial_max = 2.0 * math.sqrt(max_area_px / math.pi) * pixel_spacing[0] if max_area_px > 0 else 0.0

        finding = {
            **seg_info,
            "n_voxels": n_voxels,
            "slices_with_nodule": slices_with_nodule,
            "volume_mm3": round(volume_mm3, 2),
            "volume_cm3": round(volume_cm3, 4),
            "diameter_sphere_mm": round(diameter_sphere, 2),
            "diameter_axial_max_mm": round(diameter_axial_max, 2),
        }
        findings.append(finding)
        print(f"  Nodule {seg_num}: {diameter_axial_max:.1f}mm axial, {volume_mm3:.0f}mm³, {slices_with_nodule} coupes")

    return patient_info, exam_info, findings


# ── ÉTAPE 5: Contexte clinique ──────────────────────────────────────
def step_clinical_context(patient_id, accession):
    log(5, "📋", "CONTEXTE CLINIQUE — Chargement Excel + Classification")

    if not os.path.exists(EXCEL_PATH):
        print(f"  Excel non trouvé: {EXCEL_PATH}")
        clinical_ctx = classify_clinical_context(patient_id)
        return "", "", clinical_ctx, None

    df = pd.read_excel(EXCEL_PATH)
    patient_rows = df[df["PatientID"] == patient_id]
    print(f"  {len(patient_rows)} examens trouvés pour {patient_id}")

    # Classify clinical scenario
    clinical_ctx = classify_clinical_context(patient_id, df)
    print(f"  Scénario détecté: {clinical_ctx['scenario']} — {clinical_ctx['scenario_name']}")
    print(f"  Critères: {clinical_ctx['criteria']}")
    print(f"  Évidence: {', '.join(clinical_ctx['evidence'])}")

    current_rows = patient_rows[patient_rows["AccessionNumber"].astype(str) == str(accession)]
    previous_rows = patient_rows[patient_rows["AccessionNumber"].astype(str) != str(accession)]

    current_clinical = ""
    if not current_rows.empty:
        current_clinical = str(current_rows.iloc[0].get("Clinical information data (Pseudo reports)", ""))

    history_text = ""
    if not previous_rows.empty:
        for _, row in previous_rows.iterrows():
            history_text += f"\n- AccessionNumber: {row['AccessionNumber']}\n"
            history_text += f"  Tailles des lésions: {row.get('lesion size in mm', 'N/A')}\n"
            history_text += f"  Rapport clinique: {row.get('Clinical information data (Pseudo reports)', 'N/A')}\n"
            history_text += f"  Série SEG: {row.get('Série avec les masques de DICOM SEG', 'N/A')}\n"
        print(f"  {len(previous_rows)} examens antérieurs chargés")

    return current_clinical, history_text, clinical_ctx, df


# ── ÉTAPE 6: Rapport Gemini ─────────────────────────────────────────
def step_generate_report(patient_info, exam_info, findings, info_text,
                         current_clinical, history_text, gemini_key,
                         clinical_ctx=None, tracking_table=None,
                         recist_eval=None, audit=None, tagged_data=None,
                         vdt_text=None, reasoning_text=None):
    log(6, "🤖", "RAPPORT — Génération via Gemini")

    # Select template based on clinical scenario
    if clinical_ctx is None:
        clinical_ctx = classify_clinical_context(patient_info["id"])

    if clinical_ctx["scenario"] == "B":
        print(f"  Template: B (RECIST 1.1 — suivi oncologique)")
        prompt = build_prompt_template_b(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
            tracking_table, recist_eval, audit, tagged_data,
            vdt_text, reasoning_text,
        )
    else:
        print(f"  Template: A (Lung-RADS / Fleischner — suivi nodulaire)")
        prompt = build_prompt_template_a(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
            tracking_table, audit, tagged_data,
            vdt_text, reasoning_text,
        )

    print(f"  Prompt: {len(prompt)} caractères")

    client = genai.Client(api_key=gemini_key)

    for model_name in GEMINI_MODELS:
        try:
            print(f"  Essai modèle: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 12000},
            )
            print(f"  Succès avec {model_name}!")
            return response.text
        except Exception as e:
            err_str = str(e)
            print(f"  {model_name} échoué: {err_str[:120]}")
            continue

    raise RuntimeError("Tous les modèles Gemini ont échoué. Vérifiez votre quota/clé API.")


# ── ÉTAPE 7: Output ─────────────────────────────────────────────────
def step_output(patient_dir, patient_info, exam_info, findings, info_text, report, clinical_ctx=None):
    log(7, "💾", "OUTPUT — Sauvegarde des résultats")

    # Save summary JSON
    summary = {
        "patient": patient_info,
        "exam": exam_info,
        "findings": findings,
        "info_text": info_text,
        "clinical_context": clinical_ctx,
    }
    summary_path = os.path.join(patient_dir, "patient_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {summary_path}")

    # Save report
    report_path = os.path.join(patient_dir, "rapport_radiologique.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Rapport: {report_path}")

    return report_path


# ── Pipeline principal ───────────────────────────────────────────────
def run_pipeline(accession, gemini_key, orthanc_url=None, dry_run=False):
    """Run the full pipeline for a single AccessionNumber."""
    start_time = time.time()
    print(f"\n{'#'*60}")
    print(f"  PIPELINE UNBOXED — AccessionNumber: {accession}")
    print(f"{'#'*60}")

    patient_dir = os.path.join(PATIENTS_DIR, str(accession))
    original_dir = os.path.join(patient_dir, "original")
    results_dir = os.path.join(PATIENTS_DIR, "_results")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Discovery
    session, base_url = get_orthanc_session(orthanc_url)
    study_id, series_id, patient_id = step_discovery(session, base_url, accession)

    # 2. Download
    step_download(session, base_url, series_id, original_dir)

    # 3. Segmentation
    seg_path, info_text = step_segmentation(patient_dir, results_dir)

    # 4. Analyse SEG
    patient_info, exam_info, findings = step_analyse_seg(seg_path, original_dir)

    # 5. Contexte clinique + classification
    current_clinical, history_text, clinical_ctx, excel_df = step_clinical_context(patient_info["id"], accession)

    # 5b. Classification cible/non-cible (TÂCHE 1.2)
    findings = classify_findings(findings, clinical_ctx)
    for f in findings:
        cls = f.get("recist_classification", f.get("lungrads_classification", ""))
        print(f"  {f['SegmentLabel']}: {f['diameter_axial_max_mm']:.1f}mm → {cls}")

    # 5c. Tableau de suivi lésionnel F1↔F1 (TÂCHE 1.3)
    tracking_table = build_lesion_tracking_table(patient_info["id"], accession, findings, excel_df)
    print(f"\n{tracking_table['summary_table']}")

    # 5d. Évaluation RECIST (TÂCHE 1.3, scénario B uniquement)
    recist_eval = build_recist_evaluation(tracking_table, findings, clinical_ctx)
    if recist_eval:
        print(f"\n{recist_eval}")

    # 5e. Motif d'examen (TÂCHE 1.1)
    motif = extract_exam_motif(current_clinical)
    print(f"  Motif d'examen: {motif}")

    # 5f. Confidence scores (TÂCHE 2.1)
    pixel_sp = exam_info.get("slice_thickness", 5.0)
    try:
        ct_files_list = sorted([fn for fn in os.listdir(original_dir) if fn.endswith(".dcm")])
        ct_ds = pydicom.dcmread(os.path.join(original_dir, ct_files_list[0]))
        pixel_sp = float(ct_ds.PixelSpacing[0])
    except Exception:
        pixel_sp = 1.0
    has_excel = excel_df is not None
    print("\n  SCORES DE CONFIANCE :")
    for f in findings:
        conf = compute_confidence(f, tracking_table, pixel_sp, has_excel)
        f["confidence"] = conf
        seg_num = int(f["SegmentLabel"].replace("Finding.", ""))
        cls = f.get("recist_classification", f.get("lungrads_classification", ""))
        print(f"  F{seg_num} — {f['diameter_axial_max_mm']:.1f}mm — {cls} — "
              f"Confiance: {conf['level']} ({conf['score']}/{conf['max']})")

    # 5g. VDT calculation (TÂCHE 3.1)
    vdt_results = compute_all_vdt(tracking_table, exam_info)
    vdt_text = format_vdt_text(vdt_results)
    print(f"\n  {vdt_text}")

    # 5h. Raisonnement RECIST/Lung-RADS (TÂCHE 3.2)
    if clinical_ctx.get("scenario") == "B":
        reasoning_text = format_recist_reasoning(tracking_table, findings, clinical_ctx, recist_eval, vdt_results)
    else:
        reasoning_text = format_lungrads_reasoning(findings, tracking_table, vdt_results)
    print(f"\n{reasoning_text}")

    # 5i. Hounsfield analysis (TÂCHE 3.3 — Option B fallback)
    hu_note = analyze_hounsfield(findings, exam_info)
    print(f"\n  Hounsfield: {hu_note}")

    # 5j. Audit données manquantes (TÂCHE 2.2)
    audit = audit_data_completeness(patient_info, exam_info, findings, excel_df, patient_info["id"])
    # Add Hounsfield note to missing if not already there
    audit["missing"] = [m for m in audit["missing"] if "Densité" not in m]
    audit["missing"].append(hu_note)
    print("\n  DONNÉES MANQUANTES :")
    for m in audit["missing"]:
        print(f"  - {m}")
    for w in audit["warnings"]:
        print(f"  ⚠️ {w}")

    # 5k. Données taggées [SRC:...] (TÂCHE 2.3)
    tagged_data = format_tagged_data(
        patient_info, exam_info, findings, tracking_table,
        recist_eval, clinical_ctx, current_clinical,
    )

    # 5l. Sources trace JSON (TÂCHE 2.3)
    sources_trace = build_sources_trace(
        patient_info, exam_info, findings, tracking_table,
        recist_eval, clinical_ctx, current_clinical, seg_path,
    )
    # Add VDT to sources
    for label, v in vdt_results.items():
        if v.get("vdt_days") is not None:
            sources_trace["sources"].append({
                "data": f"VDT {label} = {v['vdt_days']}j",
                "source": "CALC",
                "method": "volume_doubling_time",
                "d1": v.get("d1"), "d2": v.get("d2"),
                "days_between": v.get("days"),
                "date_source": v.get("date_source"),
            })
    sources_path = os.path.join(patient_dir, "sources_trace.json")
    with open(sources_path, "w", encoding="utf-8") as fp:
        json.dump(sources_trace, fp, indent=2, ensure_ascii=False)
    print(f"\n  Sources trace: {sources_path}")

    if dry_run:
        print(f"\n{'='*60}")
        print("  DRY RUN — Pas d'appel Gemini. Données préparées ci-dessus.")
        print(f"{'='*60}")
        return None

    # 6. Rapport Gemini
    report = step_generate_report(
        patient_info, exam_info, findings, info_text,
        current_clinical, history_text, gemini_key, clinical_ctx,
        tracking_table, recist_eval, audit, tagged_data,
        vdt_text, reasoning_text,
    )

    # 6b. Vérification post-génération (TÂCHE 2.4)
    log("6b", "🔍", "VÉRIFICATION POST-GÉNÉRATION")
    verification = verify_report(report, findings, tracking_table, clinical_ctx, recist_eval)
    verif_path = os.path.join(patient_dir, "verification_report.json")
    with open(verif_path, "w", encoding="utf-8") as fp:
        json.dump(verification, fp, indent=2, ensure_ascii=False)

    for check in verification["checks"]:
        status = check["status"]
        icon = "✅" if status == "OK" else ("⚠️" if "NON VÉRIFIABLE" in status else "❌")
        print(f"  {icon} {check['item']}: {status}")
    print(f"  📊 Score de fiabilité : {verification['reliability_score']}")
    print(f"  💡 {verification['recommendation']}")

    # 6b2. Self-correction loop (max 1 retry)
    self_correction_info = {"applied": False, "original_issues": 0, "corrected_issues": 0}
    issues = [c for c in verification["checks"] if c["status"] != "OK"]
    self_correction_info["original_issues"] = len(issues)

    if issues:
        log("6b2", "🔄", "SELF-CORRECTION — Tentative de correction automatique")
        print(f"  {len(issues)} problème(s) détecté(s) dans le rapport initial")

        correction_prompt = build_correction_prompt(report, issues, findings)
        if correction_prompt:
            print(f"  Prompt de correction: {len(correction_prompt)} caractères")
            print(f"\n  --- INSTRUCTIONS DE CORRECTION ---")
            # Print just the instructions part
            for line in correction_prompt.split("\n"):
                if line.strip().startswith("- ERREUR") or line.strip().startswith("- ATTENTION"):
                    print(f"  {line.strip()}")
            print(f"  -----------------------------------\n")

            corrected_report, corr_model = call_gemini_correction(correction_prompt, gemini_key)
            if corrected_report:
                print(f"  Rapport corrigé reçu ({corr_model}, {len(corrected_report.split())} mots)")

                # Re-verify corrected report
                corrected_verification = verify_report(
                    corrected_report, findings, tracking_table, clinical_ctx, recist_eval
                )
                corrected_issues = [c for c in corrected_verification["checks"] if c["status"] != "OK"]
                self_correction_info["corrected_issues"] = len(corrected_issues)

                print(f"  Problèmes avant : {len(issues)} → après : {len(corrected_issues)}")

                if len(corrected_issues) <= len(issues):
                    # Correction improved or maintained quality — keep it
                    report = corrected_report
                    verification = corrected_verification
                    self_correction_info["applied"] = True

                    # Re-save verification
                    with open(verif_path, "w", encoding="utf-8") as fp:
                        json.dump(verification, fp, indent=2, ensure_ascii=False)

                    fixed = len(issues) - len(corrected_issues)
                    print(f"  ✅ Self-correction ACCEPTÉE : {fixed} problème(s) corrigé(s)")
                    for check in corrected_verification["checks"]:
                        status = check["status"]
                        icon = "✅" if status == "OK" else ("⚠️" if "NON VÉRIFIABLE" in status else "❌")
                        print(f"    {icon} {check['item']}: {status}")
                    print(f"  📊 Nouveau score de fiabilité : {corrected_verification['reliability_score']}")
                else:
                    self_correction_info["corrected_issues"] = len(corrected_issues)
                    print(f"  ⚠️ Self-correction REJETÉE (a empiré : {len(issues)} → {len(corrected_issues)} problèmes)")
            else:
                print(f"  ❌ Échec de l'appel Gemini pour la correction")
    else:
        print(f"\n  ✅ Aucun problème détecté — self-correction non nécessaire")

    # Save self-correction info in audit trail
    audit_trail_path = os.path.join(patient_dir, "audit_trail.json")
    audit_trail = {}
    if os.path.exists(audit_trail_path):
        try:
            with open(audit_trail_path) as fp:
                audit_trail = json.load(fp)
        except Exception:
            pass
    audit_trail["self_correction"] = self_correction_info
    with open(audit_trail_path, "w", encoding="utf-8") as fp:
        json.dump(audit_trail, fp, indent=2, ensure_ascii=False)

    # 6c. Auto-évaluation qualité (TÂCHE 3.4)
    log("6c", "📋", "AUTO-ÉVALUATION QUALITÉ")
    quality = evaluate_report_quality(report, findings, clinical_ctx, verification)
    quality_path = os.path.join(patient_dir, "quality_report.json")
    with open(quality_path, "w", encoding="utf-8") as fp:
        json.dump(quality, fp, indent=2, ensure_ascii=False)

    print(f"  QUALITÉ DU RAPPORT : {quality['score']}")
    for c in quality["checks"]:
        icon = "✅" if c["status"] == "OK" else "❌"
        print(f"  {icon} {c['item']}")

    # 7. Output
    report_path = step_output(patient_dir, patient_info, exam_info, findings, info_text, report, clinical_ctx)

    # ── Résumé orchestrateur ──
    elapsed = time.time() - start_time
    n_ct = len([fn for fn in os.listdir(original_dir) if fn.endswith(".dcm")])
    n_hist = len(tracking_table.get("lesions", {}).get("F1", {}).get("exams", [])) if tracking_table else 0
    conf_summary = " ".join(
        f"F{int(f['SegmentLabel'].replace('Finding.',''))}({f.get('confidence',{}).get('score','?')}/8)"
        for f in findings
    )
    vdt_summary = " ".join(
        f"{l}={v['vdt_days']:.0f}j" if v.get("vdt_days") and v["vdt_days"] != float("inf")
        else f"{l}=∞" if v.get("vdt_days") == float("inf")
        else f"{l}=N/A"
        for l, v in vdt_results.items()
    )
    recist_line = ""
    if recist_eval:
        m = re.search(r'→ ÉVALUATION : (.+)', recist_eval)
        recist_line = m.group(1) if m else ""
    n_words = len(report.split())

    print(f"\n{'#'*60}")
    print(f"  🧠 Pipeline exécuté pour {patient_info['id']} (Accession {accession})")
    print(f"  ✅ Étape 1-2 — Données CT : {n_ct} coupes téléchargées")
    print(f"  ✅ Étape 3   — Segmentation : {len(findings)} findings détectés")
    print(f"  ✅ Étape 4   — Confiance : {conf_summary}")
    print(f"  ✅ Étape 5   — Scénario : {clinical_ctx['scenario']} ({clinical_ctx['criteria']}) — \"{motif}\"")
    print(f"  ✅ Étape 6   — Tracking : {len(tracking_table.get('lesions', {}))} lésions suivies sur {n_hist} examens")
    print(f"  ✅ Étape 7   — VDT : {vdt_summary}")
    if recist_line:
        print(f"  ✅ Étape 8   — RECIST : {recist_line}")
    print(f"  ✅ Étape 9   — Données manquantes : {len(audit['missing'])} items identifiés")
    print(f"  ✅ Étape 10  — Rapport généré : {n_words} mots")
    if self_correction_info["applied"]:
        fixed = self_correction_info["original_issues"] - self_correction_info["corrected_issues"]
        print(f"  🔄 Étape 10b — Self-correction : {self_correction_info['original_issues']} → "
              f"{self_correction_info['corrected_issues']} problèmes ({fixed} hallucination(s) corrigée(s))")
    elif self_correction_info["original_issues"] > 0:
        print(f"  ⚠️ Étape 10b — Self-correction tentée mais rejetée (n'a pas amélioré le rapport)")
    print(f"  ✅ Étape 11  — Vérification : {verification['reliability_score']} vérifié, {verification['unverifiable_count']} non-vérifiable")
    print(f"  ✅ Étape 12  — Qualité : {quality['score']}")
    print(f"  📄 Fichiers sauvés :")
    print(f"     → rapport_radiologique.md")
    print(f"     → patient_summary.json")
    print(f"     → sources_trace.json")
    print(f"     → verification_report.json")
    print(f"     → quality_report.json")
    print(f"  ⏱️  Terminé en {elapsed:.1f}s")
    print(f"{'#'*60}\n")

    return report_path


def run_batch(gemini_key, orthanc_url=None):
    """Run the pipeline for all patients in the registry."""
    print("\n" + "#" * 60)
    print("  MODE BATCH — Traitement de tous les patients du registry")
    print("#" * 60)

    registry = list_entries()
    accessions = set()
    for key, entry in registry.items():
        info = entry.get("info", "")
        for line in info.split("\n"):
            if "Accession Number:" in line:
                acc = line.split("Accession Number:")[1].strip()
                if acc and acc != "0000":
                    accessions.add(acc)

    print(f"  {len(accessions)} accessions trouvées dans le registry")

    results = {}
    for i, acc in enumerate(sorted(accessions), 1):
        print(f"\n{'*'*60}")
        print(f"  BATCH [{i}/{len(accessions)}] — AccessionNumber: {acc}")
        print(f"{'*'*60}")
        try:
            report_path = run_pipeline(acc, gemini_key, orthanc_url)
            results[acc] = {"status": "success", "report": report_path}
        except Exception as e:
            print(f"  ❌ ERREUR pour {acc}: {e}")
            results[acc] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'#'*60}")
    print(f"  RÉSUMÉ BATCH")
    print(f"{'#'*60}")
    success = sum(1 for r in results.values() if r["status"] == "success")
    errors = sum(1 for r in results.values() if r["status"] == "error")
    print(f"  Succès: {success}/{len(results)}")
    print(f"  Erreurs: {errors}/{len(results)}")
    for acc, r in sorted(results.items()):
        status = "✅" if r["status"] == "success" else "❌"
        detail = r.get("report", r.get("error", ""))
        print(f"  {status} {acc}: {detail}")

    # Save batch summary
    summary_path = os.path.join(PATIENTS_DIR, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Résumé sauvegardé: {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline UNBOXED — Analyse CT + Rapport IA")
    parser.add_argument("--accession", type=str, help="AccessionNumber du patient à traiter")
    parser.add_argument("--gemini-key", type=str, help="Clé API Gemini")
    parser.add_argument("--orthanc-url", type=str, help="URL Orthanc (défaut: auto-detect)")
    parser.add_argument("--batch", action="store_true", help="Traiter tous les patients du registry")
    parser.add_argument("--dry-run", action="store_true", help="Préparer les données sans appeler Gemini")
    args = parser.parse_args()

    gemini_key = load_gemini_key(args.gemini_key)
    if not gemini_key and not args.dry_run:
        print("❌ Clé Gemini requise! Utilisez --gemini-key, GEMINI_API_KEY, ou ./work/.env")
        sys.exit(1)

    if args.batch:
        run_batch(gemini_key, args.orthanc_url)
    elif args.accession:
        run_pipeline(args.accession, gemini_key, args.orthanc_url, dry_run=args.dry_run)
    else:
        print("❌ Spécifiez --accession <NUMBER> ou --batch")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
