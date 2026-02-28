#!/usr/bin/env python3
"""
Serveur MCP — Hackathon UNBOXED
Expose les outils du pipeline radiologique comme des tools MCP.

Usage:
  python mcp_server.py
"""

import asyncio
import json
import math
import os
import sys
import time

# Ensure pipeline module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pydicom
import requests
from google import genai

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from dcm_seg_nodules import extract_seg
from dcm_seg_nodules.registry import list_entries

from pipeline import (
    classify_clinical_context,
    build_prompt_template_a,
    build_prompt_template_b,
    SCENARIO_B_KEYWORDS,
)

# ── Config ───────────────────────────────────────────────────────────
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


def _load_gemini_key():
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


def _get_orthanc():
    session = requests.Session()
    session.auth = ORTHANC_AUTH
    for url in ORTHANC_URLS:
        try:
            r = session.get(f"{url}/system", timeout=5)
            if r.status_code == 200:
                return session, url
        except Exception:
            continue
    raise ConnectionError("Cannot connect to Orthanc")


def _find_study_by_accession(session, base_url, accession):
    studies = session.get(f"{base_url}/studies").json()
    for study_id in studies:
        info = session.get(f"{base_url}/studies/{study_id}").json()
        acc = info.get("MainDicomTags", {}).get("AccessionNumber", "")
        if acc == accession:
            patient_id = info.get("PatientMainDicomTags", {}).get("PatientID", "")
            return study_id, patient_id, info
    return None, None, None


def _find_ct_series(session, base_url, study_id):
    study_data = session.get(f"{base_url}/studies/{study_id}").json()
    series_list = study_data.get("Series", [])
    best_ct = None
    best_n = 0
    target = None
    for sid in series_list:
        s_info = session.get(f"{base_url}/series/{sid}").json()
        tags = s_info.get("MainDicomTags", {})
        mod = tags.get("Modality", "")
        desc = tags.get("SeriesDescription", "")
        n = len(s_info.get("Instances", []))
        if mod == "CT":
            if "CEV" in desc.upper() and "TORAX" in desc.upper():
                target = sid
            elif n > best_n:
                best_ct = sid
                best_n = n
    return target or best_ct


def _download_series(session, base_url, series_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    series_info = session.get(f"{base_url}/series/{series_id}").json()
    instances = series_info["Instances"]
    for i, inst_id in enumerate(instances):
        resp = session.get(f"{base_url}/instances/{inst_id}/file")
        resp.raise_for_status()
        with open(os.path.join(output_dir, f"slice_{i:04d}.dcm"), "wb") as f:
            f.write(resp.content)
    return len(instances)


def _analyse_seg(seg_path, original_dir):
    seg_ds = pydicom.dcmread(seg_path)
    ct_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".dcm")])
    ct_ds = pydicom.dcmread(os.path.join(original_dir, ct_files[0]))

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

    try:
        shared = seg_ds.SharedFunctionalGroupsSequence[0]
        pm = shared.PixelMeasuresSequence[0]
        px = [float(pm.PixelSpacing[0]), float(pm.PixelSpacing[1])]
        st = float(pm.SliceThickness)
    except Exception:
        px = [float(ct_ds.PixelSpacing[0]), float(ct_ds.PixelSpacing[1])]
        st = float(ct_ds.SliceThickness)

    segments = []
    for seg in seg_ds.SegmentSequence:
        s = {
            "SegmentNumber": int(seg.SegmentNumber),
            "SegmentLabel": str(getattr(seg, "SegmentLabel", "N/A")),
        }
        segments.append(s)

    pixel_array = seg_ds.pixel_array
    frame_to_seg = {}
    if hasattr(seg_ds, "PerFrameFunctionalGroupsSequence"):
        for i, fg in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
            try:
                ref = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                frame_to_seg[i] = ref
            except Exception:
                pass

    findings = []
    for seg_info in segments:
        sn = seg_info["SegmentNumber"]
        indices = [i for i, s in frame_to_seg.items() if s == sn]
        if indices:
            frames = pixel_array[indices]
        else:
            fps = pixel_array.shape[0] // len(segments)
            start = (sn - 1) * fps
            frames = pixel_array[start:start + fps]

        n_vox = int(np.count_nonzero(frames))
        n_slices = int(np.sum(np.any(frames > 0, axis=(1, 2))))
        vol = n_vox * px[0] * px[1] * st
        d_sphere = 2.0 * (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0) if vol > 0 else 0
        max_area = max((np.count_nonzero(f) for f in frames), default=0)
        d_axial = 2.0 * math.sqrt(max_area / math.pi) * px[0] if max_area > 0 else 0

        findings.append({
            **seg_info,
            "n_voxels": n_vox,
            "slices": n_slices,
            "volume_mm3": round(vol, 2),
            "diameter_sphere_mm": round(d_sphere, 2),
            "diameter_axial_max_mm": round(d_axial, 2),
        })

    return patient_info, exam_info, findings


def _get_patient_history_from_excel(patient_id):
    if not os.path.exists(EXCEL_PATH):
        return []
    df = pd.read_excel(EXCEL_PATH)
    rows = df[df["PatientID"] == patient_id]
    result = []
    for _, row in rows.iterrows():
        result.append({
            "accession": str(row.get("AccessionNumber", "")),
            "lesion_sizes": str(row.get("lesion size in mm", "")),
            "clinical_report": str(row.get("Clinical information data (Pseudo reports)", "")),
            "seg_series": str(row.get("Série avec les masques de DICOM SEG", "")),
        })
    return result


def _generate_gemini_report(patient_info, exam_info, findings, info_text,
                            current_clinical, history_text, clinical_ctx=None):
    gemini_key = _load_gemini_key()
    if not gemini_key:
        return "ERREUR: Clé Gemini non configurée (GEMINI_API_KEY ou .env)"

    if clinical_ctx is None:
        clinical_ctx = classify_clinical_context(patient_info["id"])

    if clinical_ctx["scenario"] == "B":
        prompt = build_prompt_template_b(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
        )
    else:
        prompt = build_prompt_template_a(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
        )

    client = genai.Client(api_key=gemini_key)
    for model in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 8000},
            )
            return resp.text
        except Exception:
            continue
    return "ERREUR: Tous les modèles Gemini ont échoué."


# ══════════════════════════════════════════════════════════════════════
#  MCP Server
# ══════════════════════════════════════════════════════════════════════

server = Server("unboxed-radiology")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_patients",
            description="Liste tous les patients disponibles dans Orthanc avec PatientID, AccessionNumber et nombre de nodules connus depuis le registry.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_patient_history",
            description="Retourne l'historique complet d'un patient depuis l'Excel (tous ses examens, tailles des lésions, rapports cliniques).",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "PatientID (ex: 063F6BB9)"},
                },
                "required": ["patient_id"],
            },
        ),
        types.Tool(
            name="run_segmentation",
            description="Télécharge le CT depuis Orthanc, lance la segmentation mock, retourne les findings (nombre de nodules, tailles, volumes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "accession": {"type": "string", "description": "AccessionNumber du patient"},
                },
                "required": ["accession"],
            },
        ),
        types.Tool(
            name="generate_report",
            description="Lance le pipeline complet (download + segmentation + analyse + Gemini) et retourne le rapport radiologique en markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "accession": {"type": "string", "description": "AccessionNumber du patient"},
                },
                "required": ["accession"],
            },
        ),
        types.Tool(
            name="compare_exams",
            description="Compare tous les examens d'un patient et génère un tableau d'évolution des lésions dans le temps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "PatientID (ex: 063F6BB9)"},
                },
                "required": ["patient_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "list_patients":
            return await _tool_list_patients()
        elif name == "get_patient_history":
            return await _tool_get_patient_history(arguments["patient_id"])
        elif name == "run_segmentation":
            return await _tool_run_segmentation(arguments["accession"])
        elif name == "generate_report":
            return await _tool_generate_report(arguments["accession"])
        elif name == "compare_exams":
            return await _tool_compare_exams(arguments["patient_id"])
        else:
            return [types.TextContent(type="text", text=f"Outil inconnu: {name}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"ERREUR: {e}")]


# ── Tool implementations ────────────────────────────────────────────

async def _tool_list_patients():
    """List all patients from Orthanc + registry info."""
    session, base_url = _get_orthanc()
    studies = session.get(f"{base_url}/studies").json()

    registry = list_entries()
    # Build accession→nodule count from registry
    reg_info = {}
    for key, entry in registry.items():
        info = entry.get("info", "")
        acc = ""
        findings_count = info.count("Finding")
        for line in info.split("\n"):
            if "Accession Number:" in line:
                acc = line.split("Accession Number:")[1].strip()
        if acc:
            reg_info[acc] = {
                "nodules": findings_count,
                "description": entry.get("description", ""),
            }

    patients = []
    for study_id in studies:
        info = session.get(f"{base_url}/studies/{study_id}").json()
        tags = info.get("MainDicomTags", {})
        ptags = info.get("PatientMainDicomTags", {})
        acc = tags.get("AccessionNumber", "")
        pid = ptags.get("PatientID", "")
        pname = ptags.get("PatientName", "")
        date = tags.get("StudyDate", "")
        desc = tags.get("StudyDescription", "")

        reg = reg_info.get(acc, {})
        patients.append({
            "PatientID": pid,
            "PatientName": pname,
            "AccessionNumber": acc,
            "StudyDate": date,
            "StudyDescription": desc,
            "nodules_registry": reg.get("nodules", "unknown"),
            "registry_description": reg.get("description", "not in registry"),
        })

    result = json.dumps(patients, indent=2, ensure_ascii=False)
    return [types.TextContent(type="text", text=result)]


async def _tool_get_patient_history(patient_id: str):
    """Get patient history from Excel."""
    history = _get_patient_history_from_excel(patient_id)
    if not history:
        return [types.TextContent(type="text", text=f"Aucun historique trouvé pour {patient_id}")]
    result = json.dumps(history, indent=2, ensure_ascii=False)
    return [types.TextContent(type="text", text=result)]


async def _tool_run_segmentation(accession: str):
    """Download CT + run segmentation, return findings."""
    session, base_url = _get_orthanc()
    study_id, patient_id, _ = _find_study_by_accession(session, base_url, accession)
    if not study_id:
        return [types.TextContent(type="text", text=f"Étude {accession} non trouvée")]

    series_id = _find_ct_series(session, base_url, study_id)
    if not series_id:
        return [types.TextContent(type="text", text="Aucune série CT trouvée")]

    patient_dir = os.path.join(PATIENTS_DIR, str(accession))
    original_dir = os.path.join(patient_dir, "original")
    results_dir = os.path.join(PATIENTS_DIR, "_results")

    n = _download_series(session, base_url, series_id, original_dir)

    seg_path, info_text = extract_seg(patient_dir, output_dir=results_dir, series_subdir="original")

    patient_info, exam_info, findings = _analyse_seg(seg_path, original_dir)

    result = {
        "patient": patient_info,
        "exam": exam_info,
        "findings": findings,
        "info_text": info_text,
        "instances_downloaded": n,
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]


async def _tool_generate_report(accession: str):
    """Full pipeline: download + seg + analyse + Gemini report."""
    seg_result = await _tool_run_segmentation(accession)
    data = json.loads(seg_result[0].text)

    if "findings" not in data:
        return seg_result

    patient_info = data["patient"]
    exam_info = data["exam"]
    findings = data["findings"]
    info_text = data["info_text"]

    # Clinical context + classification
    clinical_ctx = classify_clinical_context(patient_info["id"])
    history = _get_patient_history_from_excel(patient_info["id"])
    current_clinical = ""
    history_text = ""
    for h in history:
        if h["accession"] == str(accession):
            current_clinical = h["clinical_report"]
        else:
            history_text += (
                f"\n- Accession: {h['accession']}\n"
                f"  Tailles: {h['lesion_sizes']}\n"
                f"  Rapport: {h['clinical_report']}\n"
            )

    report = _generate_gemini_report(
        patient_info, exam_info, findings, info_text,
        current_clinical, history_text, clinical_ctx,
    )

    # Save
    patient_dir = os.path.join(PATIENTS_DIR, str(accession))
    os.makedirs(patient_dir, exist_ok=True)
    report_path = os.path.join(patient_dir, "rapport_radiologique.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return [types.TextContent(type="text", text=report)]


async def _tool_compare_exams(patient_id: str):
    """Compare all exams for a patient, adapted to clinical scenario."""
    history = _get_patient_history_from_excel(patient_id)
    if not history:
        return [types.TextContent(type="text", text=f"Aucun historique pour {patient_id}")]

    clinical_ctx = classify_clinical_context(patient_id)
    scenario = clinical_ctx["scenario"]

    lines = [
        f"# Évolution des lésions — Patient {patient_id}",
        f"**Scénario: {clinical_ctx['scenario_name']}** | Critères: {clinical_ctx['criteria']}",
        "",
    ]

    # Parse all exam sizes
    exams_data = []
    for exam in history:
        size_list = [s.strip() for s in exam["lesion_sizes"].split("\n") if s.strip()]
        sizes_f = []
        for s in size_list:
            try:
                sizes_f.append(float(s))
            except ValueError:
                sizes_f.append(0.0)
        exams_data.append({"accession": exam["accession"], "sizes": sizes_f, "report": exam["clinical_report"]})

    max_lesions = max(len(e["sizes"]) for e in exams_data) if exams_data else 0

    # Build header
    header = "| AccessionNumber |"
    separator = "|-----------------|"
    for i in range(max_lesions):
        header += f" Lésion {i+1} |"
        separator += "----------|"
    if scenario == "B":
        header += " Somme (mm) |"
        separator += "------------|"
    lines.extend([header, separator])

    # Build rows
    for e in exams_data:
        row = f"| {e['accession']} |"
        for i in range(max_lesions):
            val = f"{e['sizes'][i]:.1f}mm" if i < len(e['sizes']) else "-"
            row += f" {val} |"
        if scenario == "B":
            row += f" {sum(e['sizes']):.1f} |"
        lines.append(row)

    if len(exams_data) >= 2:
        lines.append("")
        first = exams_data[0]
        last = exams_data[-1]

        if scenario == "B":
            # ── RECIST analysis ──
            lines.append("## Évaluation RECIST 1.1")
            lines.append("")
            lines.append("### Évolution par lésion cible")
            for i in range(min(len(first["sizes"]), len(last["sizes"]))):
                d1, d2 = first["sizes"][i], last["sizes"][i]
                diff = d2 - d1
                pct = (diff / d1) * 100 if d1 > 0 else 0
                trend = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"- Lésion cible {i+1}: {d1:.1f}mm → {d2:.1f}mm ({trend} {pct:+.1f}%)")

            # Sum of diameters
            sum_baseline = sum(first["sizes"])
            sum_current = sum(last["sizes"])
            sum_change = ((sum_current - sum_baseline) / sum_baseline) * 100 if sum_baseline > 0 else 0

            # Find nadir
            nadir_sum = min(sum(e["sizes"]) for e in exams_data)
            nadir_change = ((sum_current - nadir_sum) / nadir_sum) * 100 if nadir_sum > 0 else 0

            lines.append("")
            lines.append("### Somme des diamètres des lésions cibles")
            lines.append(f"- **Baseline:** {sum_baseline:.1f}mm")
            lines.append(f"- **Nadir:** {nadir_sum:.1f}mm")
            lines.append(f"- **Actuel:** {sum_current:.1f}mm")
            lines.append(f"- **Variation vs baseline:** {sum_change:+.1f}%")
            lines.append(f"- **Variation vs nadir:** {nadir_change:+.1f}%")
            lines.append("")

            # RECIST evaluation (vs nadir for PD, vs baseline for PR)
            if sum_current == 0:
                recist = "CR (Réponse Complète)"
            elif sum_change <= -30:
                recist = "PR (Réponse Partielle)"
            elif nadir_change >= 20 and (sum_current - nadir_sum) >= 5:
                recist = "PD (Progression)"
            else:
                recist = "SD (Maladie Stable)"

            lines.append(f"### **Évaluation RECIST globale: {recist}**")

        else:
            # ── Lung-RADS / Fleischner analysis ──
            lines.append("## Analyse de l'évolution (Lung-RADS / Fleischner)")
            lines.append("")
            for i in range(min(len(first["sizes"]), len(last["sizes"]))):
                d1, d2 = first["sizes"][i], last["sizes"][i]
                diff = d2 - d1
                pct = (diff / d1) * 100 if d1 > 0 else 0
                trend = "↑ croissance" if diff > 1 else "↓ régression" if diff < -1 else "→ stable"

                # Doubling time estimate (assuming ~180 days between exams as placeholder)
                dt_text = ""
                if d1 > 0 and d2 > 0 and d1 != d2:
                    v1 = (4/3) * math.pi * (d1/2)**3
                    v2 = (4/3) * math.pi * (d2/2)**3
                    if v2 > v1:
                        dt_days = (180 * math.log(2)) / math.log(v2 / v1)
                        dt_text = f" | Temps de doublement estimé: ~{dt_days:.0f} jours"

                # Lung-RADS individual
                if d2 < 6:
                    lung_rads = "LR2"
                elif d2 < 8:
                    lung_rads = "LR3"
                elif d2 < 15:
                    lung_rads = "LR4A" if diff > 1 else "LR3"
                else:
                    lung_rads = "LR4B" if diff > 1 else "LR4A"

                lines.append(
                    f"- Nodule {i+1}: {d1:.1f}mm → {d2:.1f}mm ({trend}, {pct:+.1f}%) "
                    f"| **{lung_rads}**{dt_text}"
                )

            # Global Lung-RADS = highest individual
            all_sizes = last["sizes"]
            max_size = max(all_sizes) if all_sizes else 0
            if max_size < 6:
                global_lr = "LR2 — Bénin"
                reco = "Contrôle annuel"
            elif max_size < 8:
                global_lr = "LR3 — Probablement bénin"
                reco = "Contrôle à 6 mois"
            elif max_size < 15:
                global_lr = "LR4A — Suspect"
                reco = "Contrôle à 3 mois ou PET-CT"
            else:
                global_lr = "LR4B — Très suspect"
                reco = "Biopsie ou PET-CT recommandé"

            lines.append("")
            lines.append(f"### **Lung-RADS global: {global_lr}**")
            lines.append(f"### Recommandation: {reco}")

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


# ── Run server ───────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
