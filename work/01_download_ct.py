#!/usr/bin/env python3
"""Prompt 2 — Télécharge la série CT du patient test depuis Orthanc."""

import os
import json
import requests
import pydicom

ORTHANC_URL = "https://orthanc.unboxed-2026.ovh"
AUTH = ("unboxed", "unboxed2026")
TARGET_ACCESSION = "10969511"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "patient_test", "original")

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.auth = AUTH


# 1. List all studies
print("=== Listing studies ===")
studies = session.get(f"{ORTHANC_URL}/studies").json()
print(f"Found {len(studies)} studies")

# 2. Find study with AccessionNumber = 10969511
target_study_id = None
for study_id in studies:
    info = session.get(f"{ORTHANC_URL}/studies/{study_id}").json()
    accession = info.get("MainDicomTags", {}).get("AccessionNumber", "")
    patient_id = info.get("PatientMainDicomTags", {}).get("PatientID", "")
    if accession == TARGET_ACCESSION:
        target_study_id = study_id
        print(f"Found target study: {study_id}")
        print(f"  PatientID: {patient_id}, AccessionNumber: {accession}")
        break

if not target_study_id:
    raise SystemExit(f"Study with AccessionNumber={TARGET_ACCESSION} not found!")

# 3. Find CT series "CEV torax"
print("\n=== Listing series ===")
study_data = session.get(f"{ORTHANC_URL}/studies/{target_study_id}").json()
series_list = study_data.get("Series", [])
target_series_id = None
for series_id in series_list:
    s_info = session.get(f"{ORTHANC_URL}/series/{series_id}").json()
    tags = s_info.get("MainDicomTags", {})
    modality = tags.get("Modality", "")
    desc = tags.get("SeriesDescription", "")
    n_instances = len(s_info.get("Instances", []))
    print(f"  Series: {desc} | Modality: {modality} | Instances: {n_instances}")
    if modality == "CT" and "CEV" in desc.upper():
        target_series_id = series_id
        print(f"  >>> TARGET SERIES FOUND")

if not target_series_id:
    # Fallback: pick first CT series with most instances
    for series_id in series_list:
        s_info = session.get(f"{ORTHANC_URL}/series/{series_id}").json()
        if s_info.get("MainDicomTags", {}).get("Modality") == "CT":
            target_series_id = series_id
            print(f"  >>> Fallback: using first CT series")
            break

if not target_series_id:
    raise SystemExit("No CT series found!")

# 4. Download all instances
series_info = session.get(f"{ORTHANC_URL}/series/{target_series_id}").json()
instances = series_info["Instances"]
print(f"\n=== Downloading {len(instances)} instances ===")

for i, inst_id in enumerate(instances):
    resp = session.get(f"{ORTHANC_URL}/instances/{inst_id}/file")
    resp.raise_for_status()
    filepath = os.path.join(OUTPUT_DIR, f"slice_{i:04d}.dcm")
    with open(filepath, "wb") as f:
        f.write(resp.content)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  Downloaded {i+1}/{len(instances)}")

print(f"  Done: {len(instances)} files saved to {OUTPUT_DIR}")

# 5. Read first DICOM and display metadata
print("\n=== DICOM Metadata (first slice) ===")
dcm_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".dcm")])
ds = pydicom.dcmread(os.path.join(OUTPUT_DIR, dcm_files[0]))

fields = [
    ("PatientID", "PatientID"),
    ("PatientAge", "PatientAge"),
    ("PatientSex", "PatientSex"),
    ("StudyDate", "StudyDate"),
    ("StudyDescription", "StudyDescription"),
    ("SeriesDescription", "SeriesDescription"),
    ("Modality", "Modality"),
    ("AccessionNumber", "AccessionNumber"),
    ("Manufacturer", "Manufacturer"),
    ("SliceThickness", "SliceThickness"),
    ("PixelSpacing", "PixelSpacing"),
    ("Rows", "Rows"),
    ("Columns", "Columns"),
]

for label, attr in fields:
    val = getattr(ds, attr, "N/A")
    print(f"  {label}: {val}")

print("\nDone!")
