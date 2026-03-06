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
    xs = coords[0::2]
    ys = coords[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def get_centroid(coords):
    bbox = get_bbox(coords)
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

# ==========================================
# STAGE 0: DOC INTEL & ANNOTATION
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
            
    return elements

def draw_annotations(img_array, elements):
    """Draws bounding boxes and highly visible ID tags onto the image."""
    annotated = img_array.copy()
    for el in elements:
        x, y, x_max, y_max = map(int, el["bbox"])
        cv2.rectangle(annotated, (x, y), (x_max, y_max), (255, 0, 0), 2)
        
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
    
    shapely_data = []
    for el in elements:
        pts = [(el["polygon"][i], el["polygon"][i+1]) for i in range(0, 8, 2)]
        poly = Polygon(pts)
        shapely_data.append({"element": el, "shape": poly, "area": poly.area})
        
    # Sort by area (Largest to smallest) to check biggest containers first
    shapely_data.sort(key=lambda x: x["area"], reverse=True)
    
    hierarchy = []
    for child_item in shapely_data:
        child = child_item["element"]
        child_centroid = Point(child["centroid"])
        
        parent_id = None
        # Reverse iterate to find the SMALLEST polygon that contains the point
        for possible_parent_item in reversed(shapely_data): 
            parent = possible_parent_item["element"]
            if child["id"] == parent["id"]: continue
            
            # If the child's center is inside this parent box, it belongs to it
            if possible_parent_item["shape"].contains(child_centroid):
                parent_id = parent["id"]
                break
                
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
    
    id_lookup = {el["id"]: el["content"] for el in elements}

    prompt = f"""
    You are a Cloud Architecture Router.
    I have overlaid bounding boxes with ID numbers on this diagram.
    Lookup Table: {json.dumps(id_lookup)}
    
    TASK: Trace ONLY the MACRO connections. This means lines connecting major outer boundaries, large containers, or standalone systems. 
    DO NOT trace internal micro-connections (like lines inside a subnet).
    
    Output JSON format:
    {{
      "macro_connections": [
        {{"source_id": "ID", "target_id": "ID", "flow": "One-way | Bi-directional", "style_and_meaning": "color/type (meaning)"}}
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
# STAGE 3: AGENTIC SINGLE-SHOT (ORPHAN LOOPING)
# ==========================================
def run_stage3_agentic_loop(annotated_img_array, hierarchy, macro_connections):
    print("-> [Stage 3] Agentic Loop: Tracing unmapped child nodes (Micro-routing)...")
    client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_VERSION)
    base64_image = encode_image_base64(annotated_img_array)
    
    # 1. Identify leaf nodes (components that don't have any children)
    parent_ids = {item["parent_id"] for item in hierarchy if item["parent_id"] is not None}
    leaf_nodes = [item for item in hierarchy if item["id"] not in parent_ids]
    
    # 2. Filter out leaves that were already routed in Stage 2
    routed_ids = set()
    for conn in macro_connections:
        routed_ids.add(conn["source_id"])
        routed_ids.add(conn["target_id"])
        
    orphans = [node for node in leaf_nodes if node["id"] not in routed_ids]
    
    micro_connections = []
    print(f"   -> Found {len(orphans)} unbound components. Initiating Single-Shot trace loop...")

    # NOTE: In production, limit this loop or run it asynchronously. 
    # For this script, we trace the first 5 orphans to prevent massive API wait times during testing.
    for orphan in orphans[:5]: 
        print(f"      -> Tracing line for: {orphan['content']} [ID: {orphan['id']}]")
        
        prompt = f"""
        This is a Single-Shot routing task.
        1. Find the box explicitly labeled [{orphan['id']}] ({orphan['content']}).
        2. Put all your visual focus on this single box. 
        3. There is a line physically touching or originating from this box. Follow ONLY this line.
        4. What is the ID of the box at the exact end of this line?
        
        Do not look at any other lines in the image. Do not guess based on cloud architecture. Follow the pixels.
        
        Output JSON format:
        {{
          "target_id": "ID",
          "flow": "One-way | Bi-directional",
          "style_and_meaning": "color/type"
        }}
        """
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            if "target_id" in result and result["target_id"]:
                micro_connections.append({
                    "source_id": orphan['id'],
                    "target_id": str(result["target_id"]),
                    "flow": result.get("flow", "One-way"),
                    "style_and_meaning": result.get("style_and_meaning", "Unknown")
                })
        except Exception as e:
            print(f"      -> Trace failed for [ID: {orphan['id']}]: {e}")

    return micro_connections

# ==========================================
# STAGE 4: RECURSIVE CONSOLIDATION
# ==========================================
def build_tree(node_id, id_to_node, parent_to_children):
    """Recursively builds the exact nested JSON hierarchy structure."""
    node = id_to_node[node_id]
    children_ids = parent_to_children.get(node_id, [])
    
    if not children_ids:
        # It's a leaf node, return just the string name as per expected JSON
        return node["content"]
        
    # It's a container node
    children_trees = [build_tree(child_id, id_to_node, parent_to_children) for child_id in children_ids]
    return {
        "name": node["content"],
        "type": "Container/Boundary", # Can be enriched via LLM if specific types are needed
        "children": children_trees
    }

def run_stage4_consolidation(hierarchy, macro_conns, micro_conns):
    print("-> [Stage 4] Python: Assembling nested JSON Tree...")
    
    id_to_node = {item["id"]: item for item in hierarchy}
    parent_to_children = {}
    root_nodes = []
    
    for item in hierarchy:
        pid = item["parent_id"]
        if pid is None:
            root_nodes.append(item["id"])
        else:
            if pid not in parent_to_children:
                parent_to_children[pid] = []
            parent_to_children[pid].append(item["id"])
            
    # Separate purely unbound entities (Roots with no children) from major boundaries
    architectural_hierarchy = []
    unbound_entities = []
    
    for root_id in root_nodes:
        tree = build_tree(root_id, id_to_node, parent_to_children)
        # If the tree is just a string, it means it has no children and is floating
        if isinstance(tree, str):
            unbound_entities.append(tree)
        else:
            architectural_hierarchy.append(tree)

    # Resolve Connections
    all_edges = []
    for edge in macro_conns + micro_conns:
        if edge.get("source_id") not in id_to_node or edge.get("target_id") not in id_to_node:
            continue
        all_edges.append({
            "source": id_to_node[edge["source_id"]]["content"],
            "target": id_to_node[edge["target_id"]]["content"],
            "flow": edge.get("flow", "Unknown"),
            "style_and_meaning": edge.get("style_and_meaning", "Unknown")
        })

    return {
        "architectural_hierarchy": architectural_hierarchy,
        "external_and_unbound_entities": unbound_entities,
        "end_to_end_flows": all_edges
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
        
        img_array = cv2.imread(str(img_path))
        
        # --- Stage 0 ---
        elements = run_stage0_extraction(img_path)
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
        micro_connections = run_stage3_agentic_loop(annotated_img, hierarchy, macro_connections)
        with open(OUTPUT_DIR / f"{base_name}_stage3_micro_routing_{timestamp}.json", "w") as f:
            json.dump(micro_connections, f, indent=4)
            
        # --- Stage 4 ---
        final_graph = run_stage4_consolidation(hierarchy, macro_connections, micro_connections)
        with open(OUTPUT_DIR / f"{base_name}_stage4_FINAL_GRAPH_{timestamp}.json", "w") as f:
            json.dump(final_graph, f, indent=4)
            
        print(f"-> SUCCESS: Final graph generated for {base_name}.")

if __name__ == "__main__":
    main()
