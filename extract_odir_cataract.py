"""
ODIR-5K Cataract & Balanced Normal Dataset Extractor
Extracts 402 Cataract images and 402 matched Normal images from ODIR-5K
into data/raw/cataract/ and data/raw/normal/ for CDSS training.
"""

import os
import csv
import shutil
import random

def extract_cataract_and_normal(
    csv_path="full_df.csv",
    src_images_dir=os.path.join("ODIR-5K", "ODIR-5K", "Training Images"),
    output_base="data/raw",
    seed=42
):
    print("=" * 65)
    print("ODIR-5K: Extracting Cataract and Balanced Normal Dataset")
    print("=" * 65)

    random.seed(seed)
    cataract_dir = os.path.join(output_base, "cataract")
    normal_dir = os.path.join(output_base, "normal")

    os.makedirs(cataract_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    cataract_rows = []
    normal_rows = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            
            # Verify file exists on disk
            img_path = os.path.join(src_images_dir, fname)
            if not os.path.exists(img_path):
                # Fallback to preprocessed_images if needed
                alt_path = os.path.join("preprocessed_images", fname)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    continue

            row["_resolved_path"] = img_path

            # Cataract: C == 1
            if row.get("C") == "1":
                cataract_rows.append(row)
            # Pure Normal: N == 1 and no other conditions
            elif (
                row.get("N") == "1"
                and row.get("D") == "0"
                and row.get("G") == "0"
                and row.get("A") == "0"
                and row.get("H") == "0"
                and row.get("M") == "0"
                and row.get("O") == "0"
            ):
                normal_rows.append(row)

    print(f"[*] Found {len(cataract_rows)} valid Cataract images.")
    print(f"[*] Found {len(normal_rows)} pure Normal images.")

    # Match count to ensure a balanced 1:1 dataset
    target_count = len(cataract_rows)
    selected_normal_rows = random.sample(normal_rows, min(target_count, len(normal_rows)))

    print(f"[*] Sampling {len(selected_normal_rows)} Normal images for 1:1 class balance.")

    # Copy Cataract images
    print(f"[*] Copying Cataract images to '{cataract_dir}'...")
    cataract_copied = 0
    for r in cataract_rows:
        src = r["_resolved_path"]
        dst = os.path.join(cataract_dir, r["filename"])
        shutil.copy2(src, dst)
        cataract_copied += 1

    # Copy Normal images
    print(f"[*] Copying Normal images to '{normal_dir}'...")
    normal_copied = 0
    for r in selected_normal_rows:
        src = r["_resolved_path"]
        dst = os.path.join(normal_dir, r["filename"])
        shutil.copy2(src, dst)
        normal_copied += 1

    # Create filtered metadata CSV
    meta_csv_path = os.path.join(output_base, "cataract_normal_metadata.csv")
    fieldnames = [
        "filename", "label", "patient_id", "patient_age", "patient_sex",
        "diagnostic_keywords", "original_source"
    ]
    
    with open(meta_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in cataract_rows:
            writer.writerow({
                "filename": r["filename"],
                "label": "cataract",
                "patient_id": r.get("ID", ""),
                "patient_age": r.get("Patient Age", ""),
                "patient_sex": r.get("Patient Sex", ""),
                "diagnostic_keywords": r.get("Left-Diagnostic Keywords", "") if "left" in r["filename"].lower() else r.get("Right-Diagnostic Keywords", ""),
                "original_source": r["_resolved_path"]
            })

        for r in selected_normal_rows:
            writer.writerow({
                "filename": r["filename"],
                "label": "normal",
                "patient_id": r.get("ID", ""),
                "patient_age": r.get("Patient Age", ""),
                "patient_sex": r.get("Patient Sex", ""),
                "diagnostic_keywords": r.get("Left-Diagnostic Keywords", "") if "left" in r["filename"].lower() else r.get("Right-Diagnostic Keywords", ""),
                "original_source": r["_resolved_path"]
            })

    print("=" * 65)
    print(f"[OK] Extraction Complete!")
    print(f"    - Cataract Images: {cataract_copied} in {cataract_dir}")
    print(f"    - Normal Images  : {normal_copied} in {normal_dir}")
    print(f"    - Metadata CSV   : {meta_csv_path}")
    print("=" * 65)

if __name__ == "__main__":
    extract_cataract_and_normal()
