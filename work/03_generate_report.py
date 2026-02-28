#!/usr/bin/env python3
"""Prompt 4 — Génère le rapport radiologique via Gemini."""

import os
import json
import pandas as pd
from google import genai

WORK_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(WORK_DIR)
SUMMARY_PATH = os.path.join(WORK_DIR, "patient_summary.json")
EXCEL_PATH = os.path.join(BASE_DIR, "Liste examen UNBOXED finaliseģe v2 (avec mesures).xlsx")
REPORT_PATH = os.path.join(WORK_DIR, "rapport_radiologique.md")

# 1. Load patient summary
print("=== Chargement des données ===")
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = json.load(f)

patient = summary["patient"]
exam = summary["exam"]
findings = summary["findings"]
info_text = summary["info_text"]

print(f"Patient: {patient['id']}, AccessionNumber: {exam['accession']}")

# 2. Load Excel and filter
df = pd.read_excel(EXCEL_PATH)
patient_rows = df[df["PatientID"] == patient["id"]].copy()
print(f"Examens trouvés dans Excel pour {patient['id']}: {len(patient_rows)}")

# Current exam
current_exam_rows = patient_rows[patient_rows["AccessionNumber"].astype(str) == str(exam["accession"])]
# Previous exams
previous_exam_rows = patient_rows[patient_rows["AccessionNumber"].astype(str) != str(exam["accession"])]

# Format current clinical report
current_clinical = ""
if not current_exam_rows.empty:
    current_clinical = str(current_exam_rows.iloc[0].get("Clinical information data (Pseudo reports)", ""))
    current_lesion_sizes = str(current_exam_rows.iloc[0].get("lesion size in mm", ""))

# Format history
history_text = ""
if not previous_exam_rows.empty:
    for _, row in previous_exam_rows.iterrows():
        history_text += f"\n- AccessionNumber: {row['AccessionNumber']}\n"
        history_text += f"  Tailles des lésions: {row.get('lesion size in mm', 'N/A')}\n"
        history_text += f"  Rapport clinique: {row.get('Clinical information data (Pseudo reports)', 'N/A')}\n"
        history_text += f"  Série SEG: {row.get('Série avec les masques de DICOM SEG', 'N/A')}\n"

# 3. Build findings text
findings_text = f"Nombre de nodules détectés: {len(findings)}\n"
for i, f in enumerate(findings, 1):
    findings_text += f"""
Nodule {i} ({f['SegmentLabel']}):
  - Volume: {f['volume_mm3']:.2f} mm³ ({f['volume_cm3']:.4f} cm³)
  - Diamètre sphère équivalente: {f['diameter_sphere_mm']:.2f} mm
  - Diamètre axial maximal: {f['diameter_axial_max_mm']:.2f} mm
  - Nombre de voxels: {f['n_voxels']}
  - Coupes contenant le nodule: {f['slices_with_nodule']}
  - Type algorithme: {f['SegmentAlgorithmType']}
  - Catégorie: {f['CategoryCodeMeaning']}
"""

# 4. Build the prompt
prompt = f"""Tu es un radiologue expert en imagerie thoracique spécialisé en oncologie pulmonaire.

CONSIGNES :
- FOCUS EXCLUSIF sur les LÉSIONS PULMONAIRES (nodules)
- Utilise les données antérieures pour analyser l'ÉVOLUTION des lésions
- Applique Lung-RADS et Fleischner
- Compare tailles actuelles vs antérieures
- Critères RECIST si contexte oncologique
- Rédige en français médical professionnel

=== DONNÉES PATIENT ===
- ID: {patient['id']}
- Âge: {patient['age']}
- Sexe: {patient['sex']}

=== EXAMEN ACTUEL ===
- Date: {exam['date']}
- Description: {exam['description']}
- Modalité: {exam['modality']}
- Constructeur: {exam['manufacturer']}
- Épaisseur de coupe: {exam['slice_thickness']} mm
- AccessionNumber: {exam['accession']}

=== RÉSULTATS SEGMENTATION AI ===
{findings_text}

=== INFORMATIONS DU REGISTRE ===
{info_text}

=== RAPPORT CLINIQUE ACTUEL ===
{current_clinical if current_clinical else "Non disponible"}

=== HISTORIQUE PATIENT (EXAMENS ANTÉRIEURS) ===
{history_text if history_text else "Aucun examen antérieur disponible"}

=== FORMAT DU RAPPORT DEMANDÉ ===
1. RENSEIGNEMENTS CLINIQUES
2. TECHNIQUE
3. RÉSULTATS — NODULES PULMONAIRES (pour chaque nodule : localisation estimée, dimensions actuelles, morphologie, classification Lung-RADS, comparaison avec examen antérieur si disponible)
4. SYNTHÈSE DE L'ÉVOLUTION (critères RECIST si contexte oncologique)
5. CONCLUSION (Lung-RADS global, recommandations, délai de contrôle suggéré)
6. AVERTISSEMENT IA (mentionner que ce rapport est généré par IA et doit être validé par un radiologue)
"""

print(f"\n=== Prompt construit ({len(prompt)} caractères) ===")
print("Envoi à Gemini...")

# 5. Call Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # Try loading from .env file
    env_path = os.path.join(WORK_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
if not api_key:
    raise SystemExit("GEMINI_API_KEY non définie ! Export la variable d'env ou crée ./work/.env")

client = genai.Client(api_key=api_key)

# Try multiple models in case of quota issues
models_to_try = ["gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]
response = None
for model_name in models_to_try:
    try:
        print(f"  Trying model: {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 8000},
        )
        print(f"  Success with {model_name}!")
        break
    except Exception as e:
        print(f"  {model_name} failed: {e}")
        continue

if response is None:
    raise SystemExit("All Gemini models failed. Check your API key quota.")

report = response.text
print("\n=== RAPPORT RADIOLOGIQUE GÉNÉRÉ ===\n")
print(report)

# 6. Save report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"\nRapport sauvegardé dans {REPORT_PATH}")
