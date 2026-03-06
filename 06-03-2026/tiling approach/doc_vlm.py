import os
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import cv2
import numpy as np

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, DocumentContentFormat
from openai import AzureOpenAI

load_dotenv()

DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

INPUT_DIR = Path("input")
date_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"output/pipeline_results/{date_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_image_base64(img_array):
    """Encodes an OpenCV image array to base64 for the OpenAI API."""
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# STAGE 1: AZURE DOC INTELLIGENCE (Unchanged)
# ==========================================
def run_stage1_doc_intelligence(image_path):
    print(f"-> [Stage 1] Azure OCR on {image_path.name}...")
    client = DocumentIntelligenceClient(DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY))
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", body=f, 
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=["ocrHighResolution"], output=[AnalyzeOutputOption.FIGURES]
        )
    return poller.result().as_dict()

def minify_doc_intel(raw_ocr):
    """Extracts polygons and assigns simple integer IDs."""
    elements = []
    counter = 1
    
    # Merge text and figures into one list for the VLM to reference
    for para in raw_ocr.get("paragraphs", []):
        if para.get("boundingRegions"):
            elements.append({
                "id": str(counter),
                "type": "text",
                "content": para.get("content"),
                "polygon": para["boundingRegions"][0]["polygon"]
            })
            counter += 1
            
    for fig in raw_ocr.get("figures", []):
        if fig.get("boundingRegions"):
            elements.append({
                "id": str(counter),
                "type": "icon",
                "content": "icon_or_container",
                "polygon": fig["boundingRegions"][0]["polygon"]
            })
            counter += 1
            
    return elements

# ==========================================
# STAGE 2: ANNOTATION & VLM EXTRACTION (The Fix)
# ==========================================
def create_annotated_image(image_path, spatial_elements, output_path):
    """Draws bounding boxes and highly visible ID tags onto the image."""
    img = cv2.imread(str(image_path))
    
    for el in spatial_elements:
        pts = np.array(el["polygon"], np.int32).reshape((-1, 1, 2))
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        
        # Draw a clean bounding box (Blue for text, Green for icons)
        color = (255, 0, 0) if el["type"] == "text" else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # Draw a solid background for the text ID so it's readable
        label = f"[{el['id']}]"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - label_h - 4), (x + label_w, y), (0, 0, 0), -1)
        
        # Put the ID tag in white text
        cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    cv2.imwrite(str(output_path), img)
    return img

def run_stage2_vlm_routing(annotated_img_array, spatial_elements):
    print("-> [Stage 2] VLM: Tracing connections via annotated IDs...")
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image_base64(annotated_img_array)
    
    # Provide the VLM with a lookup dictionary so it knows what the IDs represent
    id_lookup = {el["id"]: el["content"] for el in spatial_elements if el["type"] == "text"}

    prompt = f"""
    You are an expert Cloud Architecture Data Extractor.
    
    I have provided an image of an architecture diagram. 
    I have overlaid bounding boxes on all components. Each box has a numbered ID (e.g., [1], [2]).
    
    Here is the lookup table for the text IDs:
    {json.dumps(id_lookup, indent=2)}

    Your task is to visually trace EVERY line, arrow, and connector in this diagram. 
    For every line, identify which Box ID it starts at, and which Box ID it ends at.
    
    Rules:
    1. Focus heavily on arrows to determine "direction". The arrowhead is the target.
    2. Identify the line style (solid, dashed, or specific color).
    3. If a line connects an icon (green box) to text (blue box), note both IDs.
    4. Do not hallucinate connections. If a line physically connects two boxes, list it.

    Output strict JSON in this format:
    {{
      "connections": [
        {{
          "source_id": "ID number",
          "target_id": "ID number",
          "direction": "unidirectional | bidirectional | unknown",
          "line_style": "e.g., solid black, dashed blue, orange"
        }}
      ]
    }}
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.0
    )
    
    return json.loads(response.choices[0].message.content)

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue

        print(f"\n========== STARTING ROBUST PIPELINE: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem
        
        # Step 1: Doc Intel & Minification
        raw_ocr = run_stage1_doc_intelligence(img_path)
        spatial_elements = minify_doc_intel(raw_ocr)
        
        with open(OUTPUT_DIR / f"{base_name}_step1_elements_{timestamp}.json", "w") as f:
            json.dump(spatial_elements, f, indent=4)

        # Step 2: Create Annotated Image
        annotated_img_path = OUTPUT_DIR / f"{base_name}_annotated_{timestamp}.jpg"
        annotated_img = create_annotated_image(img_path, spatial_elements, annotated_img_path)
        print(f"-> Saved annotated reference image to {annotated_img_path.name}")
        
        # Step 3: VLM Extraction
        graph_data = run_stage2_vlm_routing(annotated_img, spatial_elements)
        
        # Enrichment: Map the text content back into the final JSON for readability
        lookup = {el["id"]: el["content"] for el in spatial_elements}
        for conn in graph_data.get("connections", []):
            conn["source_text"] = lookup.get(conn["source_id"], "Icon/Unknown")
            conn["target_text"] = lookup.get(conn["target_id"], "Icon/Unknown")

        final_path = OUTPUT_DIR / f"{base_name}_final_graph_{timestamp}.json"
        with open(final_path, "w") as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"-> SUCCESS: Final routing graph saved to {final_path.name}")

if __name__ == "__main__":
    main()
