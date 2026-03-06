import os
import time
import json
import base64
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Skeletonization and Graph extraction
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, DocumentContentFormat

load_dotenv()

# Configuration
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

INPUT_DIR = Path("input")
date_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"output/pipeline_results/{date_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==========================================
# STAGE 1: AZURE DOC INTELLIGENCE
# ==========================================
def run_stage1_doc_intelligence(image_path):
    print(f"-> [Stage 1] Azure OCR & Layout (High-Res) on {image_path.name}...")
    client = DocumentIntelligenceClient(DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY))
    
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=f, 
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=["ocrHighResolution"],
            output=[AnalyzeOutputOption.FIGURES]
        )
    return poller.result().as_dict()

# ==========================================
# STAGE 1b: MINIFICATION
# ==========================================
def minify_doc_intel(raw_ocr):
    print("-> [Stage 1b] Minifying OCR into core spatial polygons...")
    minified = {
        "spatial_text": [],
        "icons": []
    }

    for i, para in enumerate(raw_ocr.get("paragraphs", [])):
        minified["spatial_text"].append({
            "id": f"p_{i}",
            "text": para.get("content"),
            "polygon": para.get("boundingRegions")[0].get("polygon") if para.get("boundingRegions") else []
        })

    for j, fig in enumerate(raw_ocr.get("figures", [])):
        minified["icons"].append({
            "id": f"fig_{j}",
            "polygon": fig.get("boundingRegions")[0].get("polygon") if fig.get("boundingRegions") else []
        })
    
    return minified

# ==========================================
# STAGE 2: MASKING, BRIDGING & SKAN
# ==========================================
def create_exclusion_mask(img_shape, spatial_data):
    """Creates a mask to blot out text and complex icons."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    for item in spatial_data.get("spatial_text", []):
        pts = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
        if len(pts) > 0:
            cv2.fillPoly(mask, [pts], 255)
            
    for item in spatial_data.get("icons", []):
        pts = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
        if len(pts) > 0:
            cv2.fillPoly(mask, [pts], 255)
            
    return mask

def run_stage2_advanced_geometry(image_path, spatial_data):
    print("-> [Stage 2] Advanced CV: Masking, Morphological Closing, and Skan Tracing...")
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarize: Lines become White (255), background Black (0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Apply Exclusion Mask to erase text and icons
    exclusion_mask = create_exclusion_mask(img.shape, spatial_data)
    lines_only = cv2.bitwise_and(binary, cv2.bitwise_not(exclusion_mask))
    
    # 3. Morphological Closing (Bridge dashed lines)
    kernel = np.ones((5,5), np.uint8)
    bridged_lines = cv2.morphologyEx(lines_only, cv2.MORPH_CLOSE, kernel)
    
    # 4. Skeletonize (Reduce to 1-pixel width)
    bool_skeleton = skeletonize(bridged_lines > 0)
    
    # 5. Extract Paths with Skan
    branch_data = summarize(Skeleton(bool_skeleton))
    extracted_lines = []
    
    for _, row in branch_data.iterrows():
        # Ignore micro-noise artifacts
        if row['branch-distance'] < 15:
            continue
            
        y1, x1 = int(row['image-coord-src-0']), int(row['image-coord-src-1'])
        y2, x2 = int(row['image-coord-dst-0']), int(row['image-coord-dst-1'])
        
        extracted_lines.append({
            "start": [x1, y1],
            "end": [x2, y2],
            "euclidean_length": float(row['euclidean-distance']),
            "path_length": float(row['branch-distance'])
        })

    # 6. Arrowhead Detection (Shi-Tomasi Corners)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.05, minDistance=5)
    arrowheads = []
    
    if corners is not None:
        corners = np.intp(corners)
        for corner in corners:
            cx, cy = corner.ravel()
            # Filter: Check if corner is near an extracted line endpoint
            for line in extracted_lines:
                dist_start = math.hypot(cx - line["start"][0], cy - line["start"][1])
                dist_end = math.hypot(cx - line["end"][0], cy - line["end"][1])
                
                if dist_start < 20 or dist_end < 20:
                    arrowheads.append({"point": [int(cx), int(cy)]})
                    break 

    return {
        "lines": extracted_lines,
        "arrowheads": arrowheads
    }

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue

        print(f"\n========== STARTING PIPELINE: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem
        
        # Step 1: Azure Doc Intelligence
        raw_ocr = run_stage1_doc_intelligence(img_path)
        with open(OUTPUT_DIR / f"{base_name}_step1_raw_ocr_{timestamp}.json", "w") as f:
            json.dump(raw_ocr, f, indent=4)

        # Step 1b: Minification
        minified_spatial = minify_doc_intel(raw_ocr)
        with open(OUTPUT_DIR / f"{base_name}_step1b_minified_spatial_{timestamp}.json", "w") as f:
            json.dump(minified_spatial, f, indent=4)

        # Step 2: Advanced Geometry extraction
        geometry = run_stage2_advanced_geometry(img_path, minified_spatial)
        with open(OUTPUT_DIR / f"{base_name}_step2_advanced_geometry_{timestamp}.json", "w") as f:
            json.dump(geometry, f, indent=4)
            
        print(f"-> SUCCESS: Clean spatial data and line coordinates extracted for {base_name}.")

if __name__ == "__main__":
    main()
