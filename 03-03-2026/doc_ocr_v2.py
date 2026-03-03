import os
import time
from datetime import datetime
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

load_dotenv()

# Config
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

INPUT_DIR = Path("input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_document_intelligence(image_path):
    print(f"-> [Stage 1] Azure OCR & Layout on {image_path.name}...")
    client = DocumentIntelligenceClient(
        endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY)
    )
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", body=f, features=["ocrHighResolution"]
        )
    return poller.result().as_dict()


def minify_spatial_data(raw_doc_json):
    minified = {"spatial_text": [], "boundaries": []}

    for i, para in enumerate(raw_doc_json.get("paragraphs", [])):
        content = para.get("content")
        minified["spatial_text"].append(
            {
                "id": f"p_{i}",
                "text": content,
                "polygon": (
                    para.get("boundingRegions")[0].get("polygon")
                    if para.get("boundingRegions")
                    else []
                ),
            }
        )

    for j, figure in enumerate(raw_doc_json.get("figures", [])):
        regions = figure.get("boundingRegions", [])
        if regions:
            minified["boundaries"].append(
                {"id": f"fig_{j}", "polygon": regions[0].get("polygon")}
            )
    return minified


def apply_hough_transform(image_path, spatial_data, output_dir, base_name, timestamp):
    print("-> [Stage 2] Computer Vision: Masking Text & Applying Hough Transform...")
    
    # 1. Read Image
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Mask out text so Canny doesn't detect letters as edges
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for item in spatial_data.get("spatial_text", []):
        poly = item.get("polygon", [])
        if len(poly) == 8: # x1, y1, x2, y2, x3, y3, x4, y4
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            # Fill the text polygon with white to erase it
            cv2.fillPoly(gray, [pts], (255, 255, 255))

    # 3. Edge Detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 4. Probabilistic Hough Transform (Tuned for architectures and dashed lines)
    # maxLineGap is set high (e.g., 20) to bridge the gaps in dashed lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20)

    # 5. Visual representation for GPT-4o
    # Dim the original image so red lines pop
    overlay_img = cv2.addWeighted(img, 0.3, np.zeros(img.shape, img.dtype), 0, 0)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw bright red lines over the detected paths
            cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    hough_output_path = output_dir / f"{base_name}_hough_{timestamp}.jpg"
    cv2.imwrite(str(hough_output_path), overlay_img)
    return hough_output_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_connections(original_image_path, hough_image_path):
    print("-> [Stage 3] GPT-4o Vision: Interpreting Hough Lines & Extracting Flow...")
    llm_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    prompt = """
        You are a Principal Cloud Architect. I have provided TWO images:
        1. The original technical architecture diagram.
        2. A Computer Vision overlay where detected geometric paths are highlighted in RED.

        Your task is to extract the exact data flow and physical connections. 

        RULES FOR EXTRACTION:
        1. GROUNDING IN CV: You may ONLY extract a connection if there is a red line (or series of red lines) in the second image physically linking two components. Do not hallucinate logical connections.
        2. ORTHOGONAL LINES: Red lines that bend at 90 degrees are part of the same continuous path. Follow the red trail from source to destination.
        3. LINE JUMPS: If a red line briefly breaks to cross over another line (a line jump), treat it as a continuous path. 
        4. DIRECTIONALITY: Look at the FIRST image to find the arrowheads on the paths highlighted by the red lines. 
            - If arrowheads are on both ends, mark flow as "Bi-directional".
            - If no arrowheads exist, mark flow as "Unknown".
        5. DASHED LINES: If the original image shows the path as dashed, note this in the "style" field.

        OUTPUT FORMAT (STRICT JSON):
        {
        "connections": [
            {
            "source": "[Exact Name of Source Component]",
            "target": "[Exact Name of Target Component]",
            "flow": "One-way | Bi-directional | Unknown",
            "style": "Solid | Dashed",
            "annotation": "[Any text written directly on or next to the line, e.g., 'HTTPS' or 'Step 1']"
            }
        ]
        }
    """

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(original_image_path)}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(hough_image_path)}"}
                    }
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content)


def consolidate_architecture(image_path, spatial_data, connections):
    print("-> [Stage 4] Consolidating Knowledge Graph Foundation...")
    llm_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    prompt = f"""
        You are a Principal Cloud Architect. Consolidate the Spatial Text Boundaries and the Verified Connections into a final JSON structural map.
        
        INPUTS:
        1. Spatial Map (Ground Truth OCR): {json.dumps(spatial_data)}
        2. Verified Flow Connections: {json.dumps(connections)}
        
        CRITICAL RULES FOR CONSOLIDATION:
        1. NO HALLUCINATION: You may only list components that exist in the "Spatial Map". If a component is in the "Connections" list but missing from the Spatial Map, you must discard it.
        2. BOUNDARY INFERENCE: Use the X/Y coordinates from the Spatial Map to determine enclosure. If components sit physically under a grouping text (e.g., "VPC" or "Subnet"), nest them accordingly.
        3. IDENTICAL NODES: If two nodes share the same text but have different X/Y coordinates, append their parent boundary to their name to make them unique (e.g., "EC2_SubnetA" and "EC2_SubnetB").
        
        OUTPUT FORMAT (STRICT JSON):
        {{
        "components": [
            {{
                "id": "[Unique ID]",
                "label": "[Exact OCR Text]",
                "type": "Boundary | Service | External",
                "parent_id": "[ID of enclosing boundary, or null]"
            }}
        ],
        "relationships": [
            {{
                "source_id": "[Matches component ID]",
                "target_id": "[Matches component ID]",
                "flow": "[One-way | Bi-directional]",
                "style": "[Solid | Dashed]"
            }}
        ]
        }}
    """

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content)


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f"output/pipeline_results/{date_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            continue

        print(f"\n========== Processing: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem

        # Stage 1: Document Intelligence
        raw_ocr = run_document_intelligence(img_path)
        minified_ocr = minify_spatial_data(raw_ocr)

        # Stage 2: Hough Transform 
        hough_img_path = apply_hough_transform(img_path, minified_ocr, output_dir, base_name, timestamp)

        # Stage 3: Visual Connection Mapping
        connections = extract_connections(img_path, hough_img_path)

        step3_path = output_dir / f"{base_name}_step3_connections_{timestamp}.json"
        with open(step3_path, "w") as f:
            json.dump(connections, f, indent=4)

        # Stage 4: Logical Consolidation
        final_json = consolidate_architecture(img_path, minified_ocr, connections)

        step4_path = output_dir / f"{base_name}_step4_final_{timestamp}.json"
        with open(step4_path, "w") as f:
            json.dump(final_json, f, indent=4)

        print(f"-> SUCCESS: Files saved to {output_dir}")

if __name__ == "__main__":
    main()
