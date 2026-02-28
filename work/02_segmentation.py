#!/usr/bin/env python3
"""Prompt 3 — Lance la segmentation mock et analyse le fichier DICOM SEG."""

import os
import json
import math
import numpy as np
import pydicom

WORK_DIR = os.path.dirname(__file__)
PATIENT_DIR = os.path.join(WORK_DIR, "patient_test")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
ORIGINAL_DIR = os.path.join(PATIENT_DIR, "original")

# 1. Run mock segmentation
print("=== Étape 1 : Segmentation mock ===")
from dcm_seg_nodules import extract_seg

seg_path, info_text = extract_seg(
    PATIENT_DIR,
    output_dir=RESULTS_DIR,
    series_subdir="original"
)
print(f"seg_path: {seg_path}")
print(f"info_text:\n{info_text}")

# 2. Read patient metadata from first CT slice
ct_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith(".dcm")])
ct_ds = pydicom.dcmread(os.path.join(ORIGINAL_DIR, ct_files[0]))
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

# 3. Read and analyze the SEG file
print("\n=== Étape 2 : Analyse du fichier SEG ===")
seg_ds = pydicom.dcmread(seg_path)

# Extract pixel spacing and slice thickness from SharedFunctionalGroupsSequence
pixel_spacing = None
seg_slice_thickness = None
try:
    shared = seg_ds.SharedFunctionalGroupsSequence[0]
    pm = shared.PixelMeasuresSequence[0]
    pixel_spacing = [float(pm.PixelSpacing[0]), float(pm.PixelSpacing[1])]
    seg_slice_thickness = float(pm.SliceThickness)
    print(f"PixelSpacing from SEG: {pixel_spacing}")
    print(f"SliceThickness from SEG: {seg_slice_thickness}")
except Exception as e:
    print(f"Warning: Could not read pixel measures from SEG: {e}")
    pixel_spacing = [float(ct_ds.PixelSpacing[0]), float(ct_ds.PixelSpacing[1])]
    seg_slice_thickness = float(ct_ds.SliceThickness)
    print(f"Using CT values: PixelSpacing={pixel_spacing}, SliceThickness={seg_slice_thickness}")

# Extract segments info
print("\n--- Segments ---")
segments = []
for seg in seg_ds.SegmentSequence:
    seg_info = {
        "SegmentNumber": int(seg.SegmentNumber),
        "SegmentLabel": str(getattr(seg, "SegmentLabel", "N/A")),
        "SegmentDescription": str(getattr(seg, "SegmentDescription", "N/A")),
        "SegmentAlgorithmType": str(getattr(seg, "SegmentAlgorithmType", "N/A")),
        "SegmentAlgorithmName": str(getattr(seg, "SegmentAlgorithmName", "N/A")),
    }
    # Category
    try:
        cat = seg.SegmentedPropertyCategoryCodeSequence[0]
        seg_info["CategoryCodeMeaning"] = str(cat.CodeMeaning)
    except Exception:
        seg_info["CategoryCodeMeaning"] = "N/A"
    # Type
    try:
        typ = seg.SegmentedPropertyTypeCodeSequence[0]
        seg_info["TypeCodeMeaning"] = str(typ.CodeMeaning)
    except Exception:
        seg_info["TypeCodeMeaning"] = "N/A"

    segments.append(seg_info)
    print(f"  Segment {seg_info['SegmentNumber']}: {seg_info['SegmentLabel']} "
          f"({seg_info['SegmentDescription']}) — {seg_info['SegmentAlgorithmType']}")

# 4. Analyze pixel data per segment
print("\n--- Analyse voxels par segment ---")
pixel_array = seg_ds.pixel_array
print(f"pixel_array shape: {pixel_array.shape}")

n_segments = len(segments)
findings = []

# pixel_array can be (n_frames, rows, cols) — frames are grouped per segment
# We need to figure out which frames belong to which segment
# Use PerFrameFunctionalGroupsSequence to map frames to segments
frame_to_segment = {}
if hasattr(seg_ds, "PerFrameFunctionalGroupsSequence"):
    for i, frame_fg in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        try:
            seg_id_seq = frame_fg.SegmentIdentificationSequence[0]
            ref_seg_num = int(seg_id_seq.ReferencedSegmentNumber)
            frame_to_segment[i] = ref_seg_num
        except Exception:
            pass

for seg_info in segments:
    seg_num = seg_info["SegmentNumber"]
    # Get frames for this segment
    frame_indices = [i for i, sn in frame_to_segment.items() if sn == seg_num]

    if frame_indices:
        seg_frames = pixel_array[frame_indices]
    else:
        # Fallback: assume frames are evenly split
        frames_per_seg = pixel_array.shape[0] // n_segments
        start = (seg_num - 1) * frames_per_seg
        end = start + frames_per_seg
        seg_frames = pixel_array[start:end]

    n_voxels = int(np.count_nonzero(seg_frames))
    # Count slices containing the nodule
    slices_with_nodule = int(np.sum(np.any(seg_frames > 0, axis=(1, 2))))

    # Compute measurements
    voxel_vol = pixel_spacing[0] * pixel_spacing[1] * seg_slice_thickness
    volume_mm3 = n_voxels * voxel_vol
    volume_cm3 = volume_mm3 / 1000.0

    # Equivalent sphere diameter
    if volume_mm3 > 0:
        diameter_sphere = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    else:
        diameter_sphere = 0.0

    # Max axial diameter: find the frame with max area, compute diameter
    max_area_px = 0
    for frame in seg_frames:
        area_px = np.count_nonzero(frame)
        if area_px > max_area_px:
            max_area_px = area_px
    if max_area_px > 0:
        diameter_axial_max = 2.0 * math.sqrt(max_area_px / math.pi) * pixel_spacing[0]
    else:
        diameter_axial_max = 0.0

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

    print(f"\n  Segment {seg_num} — {seg_info['SegmentLabel']}:")
    print(f"    Voxels non-zéro: {n_voxels}")
    print(f"    Coupes avec nodule: {slices_with_nodule}")
    print(f"    Volume: {volume_mm3:.2f} mm³ ({volume_cm3:.4f} cm³)")
    print(f"    Diamètre sphère équivalente: {diameter_sphere:.2f} mm")
    print(f"    Diamètre axial max: {diameter_axial_max:.2f} mm")

# 5. Save summary JSON
summary = {
    "patient": patient_info,
    "exam": exam_info,
    "findings": findings,
    "info_text": info_text,
}

summary_path = os.path.join(WORK_DIR, "patient_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n=== Summary saved to {summary_path} ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
