import os
import time
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from shapely.geometry import Polygon, Point
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, DocumentContentFormat
from openai import AzureOpenAI

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
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
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def get_bbox(coords):
    """Convert flat 8-point array to bounding box [x_min, y_min, x_max, y_max]."""
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def get_centroid(coords):
    """Get the center point of a polygon for robust containment checks."""
    bbox = get_bbox(coords)
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

# ==========================================
# STAGE 0: DOC INTEL & MINIFICATION
# ==========================================
def run_stage0_extraction(image_path):
    print(f"-> [Stage 0] Azure OCR & Minification on {image_path.name}...")
    client = DocumentIntelligenceClient(DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY))
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", body=f, 
            output_content_format=DocumentContentFormat.MARKDOWN,
            features=["ocrHighResolution"], output=[AnalyzeOutputOption.FIGURES]
        )
    raw_ocr = poller.result().as_dict()
    
    elements = []
    counter = 1
    for para in raw_ocr.get("paragraphs", []):
        if para.get("boundingRegions"):
            coords = para["boundingRegions"][0]["polygon"]
            elements.append({
                "id": str(counter), "type": "text", "content": para.get("content"),
                "polygon": coords, "bbox": get_bbox(coords), "centroid": get_centroid(coords)
            })
            counter += 1
            
    for fig in raw_ocr.get("figures", []):
        if fig.get("boundingRegions"):
            coords = fig["boundingRegions"][0]["polygon"]
            elements.append({
                "id": str(counter), "type": "icon", "content": "icon_or_container",
                "polygon": coords, "bbox": get_bbox(coords), "centroid": get_centroid(coords)
            })
            counter += 1
            
    return elements

def draw_annotations(img_array, elements):
    """Draws bounding boxes and highly visible ID tags onto the image."""
    annotated = img_array.copy()
    for el in elements:
        x, y, x_max, y_max = map(int, el["bbox"])
        color = (255, 0, 0) if el["type"] == "text" else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x_max, y_max), color, 2)
        
        label = f"[{el['id']}]"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, y - label_h - 4), (x + label_w, y), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated

# ==========================================
# STAGE 1: PYTHON SPATIAL HIERARCHY
# ==========================================
def run_stage1_spatial_hierarchy(elements):
    print("-> [Stage 1] Shapely: Computing Parent-Child Hierarchies deterministically...")
    
    # Convert element polygons to Shapely Polygons and calculate area
    shapely_data = []
    for el in elements:
        pts = [(el["polygon"][i], el["polygon"][i+1]) for i in range(0, 8, 2)]
        poly = Polygon(pts)
        shapely_data.append({"element": el, "shape": poly, "area": poly.area})
        
    # Sort by area (Largest to smallest) so we check biggest containers first
    shapely_data.sort(key=lambda x: x["area"], reverse=True)
    
    hierarchy = []
    for child_item in shapely_data:
        child = child_item["element"]
        child_centroid = Point(child["centroid"])
        
        parent_id = None
        # Find the smallest polygon that contains this child's centroid
        for possible_parent_item in reversed(shapely_data): 
            parent = possible_parent_item["element"]
            if child["id"] == parent["id"]: continue
            
            if possible_parent_item["shape"].contains(child_centroid):
                parent_id = parent["id"]
                break # We found the tightest bounding box because we iterate reversed (smallest first)
                
        hierarchy.append({
            "id": child["id"],
            "type": child["type"],
            "content": child["content"],
            "parent_id": parent_id
        })
        
    return hierarchy

# ==========================================
# STAGE 2: MACRO-ROUTING (GLOBAL LLM)
# ==========================================
def run_stage2_macro_routing(annotated_img_array, elements):
    print("-> [Stage 2] VLM: Extracting Macro (Container-to-Container) connections...")
    client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_VERSION)
    base64_image = encode_image_base64(annotated_img_array)
    
    id_lookup = {el["id"]: el["content"] for el in elements if el["type"] == "text"}

    prompt = f"""
    You are a Cloud Architecture Router.
    I have overlaid bounding boxes with ID numbers on this diagram.
    Lookup Table: {json.dumps(id_lookup)}
    
    TASK: Trace ONLY the MACRO connections. This means lines connecting major outer boundaries, large containers, or standalone systems. 
    DO NOT trace internal micro-connections (like lines inside a subnet).
    
    Output JSON format:
    {{
      "macro_connections": [
        {{"source_id": "ID", "target_id": "ID", "direction": "unidirectional | bidirectional", "line_style": "color/type"}}
      ]
    }}
    """
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content).get("macro_connections", [])
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        return []

# ==========================================
# STAGE 3: TARGETED MICRO-TILING (SEMANTIC ZOOM)
# ==========================================
def run_stage3_micro_tiling(original_img_array, annotated_img_array, elements, macro_connections, base_name, timestamp):
    print("-> [Stage 3] Semantic Zooming: Cropping dense regions for micro-tracing...")
    client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_VERSION)
    
    element_map = {el["id"]: el for el in elements}
    micro_connections = []
    
    # Identify unique container IDs that have connections
    containers_to_inspect = set()
    for conn in macro_connections:
        containers_to_inspect.add(conn["source_id"])
        containers_to_inspect.add(conn["target_id"])
        
    for container_id in containers_to_inspect:
        if container_id not in element_map: continue
        container_bbox = element_map[container_id]["bbox"]
        
        # Crop the annotated image to this specific container with a 40px margin
        x_min = max(0, int(container_bbox[0]) - 40)
        y_min = max(0, int(container_bbox[1]) - 40)
        x_max = min(annotated_img_array.shape[1], int(container_bbox[2]) + 40)
        y_max = min(annotated_img_array.shape[0], int(container_bbox[3]) + 40)
        
        crop_img = annotated_img_array[y_min:y_max, x_min:x_max]
        crop_path = OUTPUT_DIR / f"{base_name}_stage3_crop_container_{container_id}_{timestamp}.jpg"
        cv2.imwrite(str(crop_path), crop_img)
        
        # Find all IDs physically inside this crop for the lookup table
        children_ids = {el["id"]: el["content"] for el in elements 
                        if el["centroid"][0] >= x_min and el["centroid"][0] <= x_max
                        and el["centroid"][1] >= y_min and el["centroid"][1] <= y_max and el["type"] == "text"}
                        
        if not children_ids: continue # Empty container

        print(f"   -> Zooming into Container [{container_id}]. Found {len(children_ids)} internal elements. Querying VLM...")
        base64_crop = encode_image_base64(crop_img)
        
        prompt = f"""
        You are looking at a zoomed-in, high-resolution crop of a dense architecture diagram.
        Valid IDs in this crop: {json.dumps(children_ids)}
        
        TASK: Trace the MICRO-CONNECTIONS inside this specific crop. For example, lines connecting specific Private Endpoints to specific databases.
        
        Output JSON format:
        {{
          "micro_connections": [
            {{"source_id": "ID", "target_id": "ID", "direction": "unidirectional | bidirectional", "line_style": "color/type"}}
          ]
        }}
        """
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_crop}"}}]}],
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content).get("micro_connections", [])
            micro_connections.extend(result)
        except Exception as e:
            print(f"   -> Crop VLM failed for {container_id}: {e}")
            
    return micro_connections

# ==========================================
# STAGE 4: CONSOLIDATION
# ==========================================
def run_stage4_consolidation(hierarchy, macro_conns, micro_conns):
    print("-> [Stage 4] Consolidating into Final Knowledge Graph...")
    
    # Enrichment: Add names back to connections for readability
    id_to_name = {node["id"]: node["content"] for node in hierarchy}
    
    all_edges = []
    
    for edge in macro_conns + micro_conns:
        # Prevent appending empty or broken edges
        if edge.get("source_id") not in id_to_name or edge.get("target_id") not in id_to_name:
            continue
            
        all_edges.append({
            "source_id": edge["source_id"],
            "source_text": id_to_name.get(edge["source_id"]),
            "target_id": edge["target_id"],
            "target_text": id_to_name.get(edge["target_id"]),
            "direction": edge.get("direction", "unknown"),
            "line_style": edge.get("line_style", "unknown")
        })

    # Deduplicate edges (in case macro and micro caught the same line)
    seen = set()
    unique_edges = []
    for edge in all_edges:
        sig = f"{edge['source_id']}-{edge['target_id']}"
        if sig not in seen:
            seen.add(sig)
            unique_edges.append(edge)

    return {
        "architectural_hierarchy": hierarchy,
        "end_to_end_flows": unique_edges
    }


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue

        print(f"\n========== STARTING FAILSAFE PIPELINE: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem
        
        img_array = cv2.imread(str(img_path))
        
        # --- Stage 0 ---
        elements = run_stage0_extraction(img_path)
        with open(OUTPUT_DIR / f"{base_name}_stage0_elements_{timestamp}.json", "w") as f:
            json.dump(elements, f, indent=4)
            
        annotated_img = draw_annotations(img_array, elements)
        cv2.imwrite(str(OUTPUT_DIR / f"{base_name}_stage0_annotated_{timestamp}.jpg"), annotated_img)

        # --- Stage 1 ---
        hierarchy = run_stage1_spatial_hierarchy(elements)
        with open(OUTPUT_DIR / f"{base_name}_stage1_hierarchy_{timestamp}.json", "w") as f:
            json.dump(hierarchy, f, indent=4)
            
        # --- Stage 2 ---
        macro_connections = run_stage2_macro_routing(annotated_img, elements)
        with open(OUTPUT_DIR / f"{base_name}_stage2_macro_routing_{timestamp}.json", "w") as f:
            json.dump(macro_connections, f, indent=4)
            
        # --- Stage 3 ---
        micro_connections = run_stage3_micro_tiling(img_array, annotated_img, elements, macro_connections, base_name, timestamp)
        with open(OUTPUT_DIR / f"{base_name}_stage3_micro_routing_{timestamp}.json", "w") as f:
            json.dump(micro_connections, f, indent=4)
            
        # --- Stage 4 ---
        final_graph = run_stage4_consolidation(hierarchy, macro_connections, micro_connections)
        with open(OUTPUT_DIR / f"{base_name}_stage4_FINAL_GRAPH_{timestamp}.json", "w") as f:
            json.dump(final_graph, f, indent=4)
            
        print(f"-> SUCCESS: Complete Failsafe Extraction saved to {OUTPUT_DIR.name}")

if __name__ == "__main__":
    main()
