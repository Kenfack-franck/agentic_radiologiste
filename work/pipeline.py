#!/usr/bin/env python3
"""
Pipeline complet agentique — Hackathon UNBOXED
Usage:
  python pipeline.py --accession 10969511 --gemini-key "AIza..."
  python pipeline.py --batch --gemini-key "AIza..."
"""

import argparse
import json
import math
import os
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


# ── Prompt Templates ────────────────────────────────────────────────
def build_prompt_template_a(patient_info, exam_info, findings, info_text,
                            current_clinical, history_text, clinical_ctx):
    """Template A: Nodule follow-up (Lung-RADS / Fleischner)."""
    findings_text = _format_findings(findings)
    return f"""Tu es un radiologue expert en imagerie thoracique.

CONTEXTE CLINIQUE : {clinical_ctx['scenario_name']}
CRITÈRES À APPLIQUER : {clinical_ctx['criteria']}
INDICATION : {clinical_ctx['clinical_indication']}

CONSIGNES STRICTES :
- FOCUS EXCLUSIF sur les NODULES PULMONAIRES et leur CARACTÉRISATION
- Pour chaque nodule, attribue une classification Lung-RADS individuelle :
  * LR1 : Négatif (pas de nodule)
  * LR2 : Bénin (< 6mm solide, calcifié, stable > 2 ans)
  * LR3 : Probablement bénin (6-8mm solide, suivi à 6 mois)
  * LR4A : Suspect (> 8mm solide, > 6mm verre dépoli, nouveau, croissance lente)
  * LR4B : Très suspect (> 15mm, croissance rapide, morphologie maligne)
  * LR4X : 4A/4B avec features additionnels suspects
- Compare avec les examens antérieurs pour évaluer la STABILITÉ ou CROISSANCE
- Si mesures antérieures disponibles, calcule le TEMPS DE DOUBLEMENT :
  DT = (Δt × ln2) / ln(V2/V1) où V = 4/3 π (d/2)³
- Applique les recommandations FLEISCHNER pour le suivi
- Rédige en français médical professionnel

=== DONNÉES PATIENT ===
- ID: {patient_info['id']}
- Âge: {patient_info['age']}
- Sexe: {patient_info['sex']}

=== EXAMEN ACTUEL ===
- Date: {exam_info['date']}
- Description: {exam_info['description']}
- Modalité: {exam_info['modality']}
- Constructeur: {exam_info['manufacturer']}
- Épaisseur de coupe: {exam_info['slice_thickness']} mm
- AccessionNumber: {exam_info['accession']}

=== RÉSULTATS SEGMENTATION AI ===
{findings_text}

=== INFORMATIONS DU REGISTRE (RÉFÉRENCE — NE PAS UTILISER COMME MESURES ACTUELLES) ===
⚠️ Les diamètres mentionnés ci-dessous sont des valeurs de RÉFÉRENCE DU REGISTRE (examens précédents).
Les MESURES ACTUELLES sont UNIQUEMENT celles de la section "RÉSULTATS SEGMENTATION AI" ci-dessus.
{info_text}

=== RAPPORT CLINIQUE ACTUEL ===
{current_clinical if current_clinical else "Non disponible"}

=== HISTORIQUE PATIENT (EXAMENS ANTÉRIEURS) ===
{history_text if history_text else "Aucun examen antérieur disponible"}

=== FORMAT DU RAPPORT (sois concis, max 1500 mots) ===
1. RENSEIGNEMENTS CLINIQUES (indication, contexte de surveillance nodulaire)
2. TECHNIQUE (modalité, paramètres)
3. RÉSULTATS — NODULES PULMONAIRES
   Pour chaque nodule :
   - Localisation estimée
   - Dimensions actuelles (utilise UNIQUEMENT les mesures de la SEGMENTATION AI)
   - Morphologie (solide/subsolide/verre dépoli si déductible)
   - Classification Lung-RADS individuelle avec justification
   - Comparaison avec examen(s) antérieur(s) : stabilité, croissance, temps de doublement
4. SYNTHÈSE (nombre total de nodules, Lung-RADS global, critères Fleischner applicables)
5. CONCLUSION
   - Classification Lung-RADS globale (catégorie la plus élevée)
   - Recommandation de suivi selon Fleischner
   - Délai du prochain contrôle suggéré
   - Indication de biopsie si pertinent
6. AVERTISSEMENT IA (ce rapport est généré par IA et doit être validé par un radiologue)
"""


def build_prompt_template_b(patient_info, exam_info, findings, info_text,
                            current_clinical, history_text, clinical_ctx):
    """Template B: Oncology follow-up (RECIST 1.1)."""
    findings_text = _format_findings(findings)

    # Pre-compute RECIST sum for the prompt
    current_sum = sum(f["diameter_axial_max_mm"] for f in findings)

    return f"""Tu es un radiologue expert en imagerie thoracique spécialisé en oncologie pulmonaire.

CONTEXTE CLINIQUE : {clinical_ctx['scenario_name']}
CRITÈRES À APPLIQUER : {clinical_ctx['criteria']}
INDICATION : {clinical_ctx['clinical_indication']}
ÉLÉMENTS DÉCLENCHEURS : {', '.join(clinical_ctx['evidence'])}

CONSIGNES STRICTES :
- Ce patient a un CANCER CONFIRMÉ en cours de traitement/essai clinique
- Applique RIGOUREUSEMENT les critères RECIST 1.1 :
  * CR (Réponse Complète) : disparition de toutes les lésions cibles
  * PR (Réponse Partielle) : diminution ≥ 30% de la somme des diamètres vs baseline
  * SD (Maladie Stable) : ni PR ni PD
  * PD (Progression) : augmentation ≥ 20% de la somme des diamètres vs nadir + augmentation absolue ≥ 5mm, OU nouvelles lésions
- Identifie les lésions CIBLES (target, max 5, mesurables ≥ 10mm) et NON-CIBLES
- Compare avec le BASELINE et le NADIR (meilleure réponse)
- Somme actuelle des diamètres des lésions cibles (SEGMENTATION AI) : {current_sum:.1f} mm
- IMPORTANT : les diamètres ACTUELS sont ceux de la SEGMENTATION AI, pas ceux du registre
- Signale toute NOUVELLE lésion (= progression automatique)
- Rédige en français médical professionnel, ton précis et quantitatif

=== DONNÉES PATIENT ===
- ID: {patient_info['id']}
- Âge: {patient_info['age']}
- Sexe: {patient_info['sex']}

=== EXAMEN ACTUEL ===
- Date: {exam_info['date']}
- Description: {exam_info['description']}
- Modalité: {exam_info['modality']}
- Constructeur: {exam_info['manufacturer']}
- Épaisseur de coupe: {exam_info['slice_thickness']} mm
- AccessionNumber: {exam_info['accession']}

=== RÉSULTATS SEGMENTATION AI ===
{findings_text}

=== INFORMATIONS DU REGISTRE (RÉFÉRENCE — NE PAS UTILISER COMME MESURES ACTUELLES) ===
⚠️ Les diamètres mentionnés ci-dessous sont des valeurs de RÉFÉRENCE DU REGISTRE (examens précédents).
Les MESURES ACTUELLES sont UNIQUEMENT celles de la section "RÉSULTATS SEGMENTATION AI" ci-dessus.
{info_text}

=== RAPPORT CLINIQUE ACTUEL ===
{current_clinical if current_clinical else "Non disponible"}

=== HISTORIQUE PATIENT (EXAMENS ANTÉRIEURS) ===
{history_text if history_text else "Aucun examen antérieur disponible"}

=== FORMAT DU RAPPORT (sois concis, max 1500 mots) ===
1. RENSEIGNEMENTS CLINIQUES (indication oncologique, contexte de traitement/essai)
2. TECHNIQUE (modalité, paramètres)
3. RÉSULTATS — LÉSIONS PULMONAIRES
   a. LÉSIONS CIBLES (target lesions) :
      Pour chaque lésion cible :
      - Localisation
      - Diamètre axial actuel (mm)
      - Diamètre à l'examen précédent / baseline
      - Variation en mm et en %
   b. Somme des diamètres des lésions cibles :
      - Actuel vs baseline
      - Actuel vs nadir (meilleure réponse)
      - Variation en %
   c. LÉSIONS NON-CIBLES : présentes/absentes/en progression
   d. NOUVELLES LÉSIONS : oui/non
4. ÉVALUATION RECIST 1.1
   - Réponse des lésions cibles : CR / PR / SD / PD
   - Réponse des lésions non-cibles : CR / non-CR/non-PD / PD
   - Nouvelles lésions : oui / non
   - RÉPONSE GLOBALE : CR / PR / SD / PD
5. CONCLUSION
   - Évaluation RECIST globale
   - Tendance évolutive (amélioration / stabilité / progression)
   - Recommandation pour la suite de la prise en charge
6. AVERTISSEMENT IA (ce rapport est généré par IA et doit être validé par un radiologue)
"""


def _format_findings(findings):
    text = f"Nombre de nodules détectés: {len(findings)}\n"
    for i, f in enumerate(findings, 1):
        text += f"""
Nodule {i} ({f['SegmentLabel']}):
  - Diamètre axial maximal (MESURE ACTUELLE): {f['diameter_axial_max_mm']:.2f} mm
  - Diamètre sphère équivalente: {f['diameter_sphere_mm']:.2f} mm
  - Volume: {f['volume_mm3']:.2f} mm³ ({f['volume_cm3']:.4f} cm³)
  - Coupes contenant le nodule: {f['slices_with_nodule']}
"""
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
        return "", "", clinical_ctx

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

    return current_clinical, history_text, clinical_ctx


# ── ÉTAPE 6: Rapport Gemini ─────────────────────────────────────────
def step_generate_report(patient_info, exam_info, findings, info_text,
                         current_clinical, history_text, gemini_key,
                         clinical_ctx=None):
    log(6, "🤖", "RAPPORT — Génération via Gemini")

    # Select template based on clinical scenario
    if clinical_ctx is None:
        clinical_ctx = classify_clinical_context(patient_info["id"])

    if clinical_ctx["scenario"] == "B":
        print(f"  Template: B (RECIST 1.1 — suivi oncologique)")
        prompt = build_prompt_template_b(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
        )
    else:
        print(f"  Template: A (Lung-RADS / Fleischner — suivi nodulaire)")
        prompt = build_prompt_template_a(
            patient_info, exam_info, findings, info_text,
            current_clinical, history_text, clinical_ctx,
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
def run_pipeline(accession, gemini_key, orthanc_url=None):
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
    current_clinical, history_text, clinical_ctx = step_clinical_context(patient_info["id"], accession)

    # 6. Rapport Gemini (template adapté au scénario)
    report = step_generate_report(
        patient_info, exam_info, findings, info_text,
        current_clinical, history_text, gemini_key, clinical_ctx,
    )

    # 7. Output
    report_path = step_output(patient_dir, patient_info, exam_info, findings, info_text, report, clinical_ctx)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE TERMINÉ en {elapsed:.1f}s")
    print(f"  Patient: {patient_info['id']} | Accession: {accession}")
    print(f"  Nodules: {len(findings)} | Rapport: {report_path}")
    print(f"{'='*60}\n")

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
    args = parser.parse_args()

    gemini_key = load_gemini_key(args.gemini_key)
    if not gemini_key:
        print("❌ Clé Gemini requise! Utilisez --gemini-key, GEMINI_API_KEY, ou ./work/.env")
        sys.exit(1)

    if args.batch:
        run_batch(gemini_key, args.orthanc_url)
    elif args.accession:
        run_pipeline(args.accession, gemini_key, args.orthanc_url)
    else:
        print("❌ Spécifiez --accession <NUMBER> ou --batch")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
