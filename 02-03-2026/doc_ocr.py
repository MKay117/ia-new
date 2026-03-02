# # import os
# # import json
# # import time
# # from datetime import datetime
# # from pathlib import Path
# # from azure.core.credentials import AzureKeyCredential
# # from azure.ai.documentintelligence import DocumentIntelligenceClient
# # from azure.ai.documentintelligence.models import AnalyzeResult

# # # Configuration [cite: 27, 51]
# # ENDPOINT = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
# # KEY = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
# # INPUT_DIR = Path("input")
# # OUTPUT_BASE_DIR = Path("output/docu_intelligence")

# # def process_architecture_diagrams():
# #     # Instantiate client [cite: 27]
# #     client = DocumentIntelligenceClient(ENDPOINT, AzureKeyCredential(KEY))

# #     # Ensure output directory exists
# #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# #     if not INPUT_DIR.exists():
# #         print(f"Input directory {INPUT_DIR} not found.")
# #         return

# #     for file_path in INPUT_DIR.iterdir():
# #         if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf', '.tiff']:
# #             print(f"Processing: {file_path.name}...")

# #             with open(file_path, "rb") as f:
# #                 # Step 1: Extract Layout (Text, Tables, Selection Marks, Polygons) [cite: 51, 52]
# #                 poller = client.begin_analyze_document("prebuilt-layout", body=f)
# #                 result: AnalyzeResult = poller.result() # Wait for long-running operation

# #             # Convert to JSON-serializable dictionary
# #             analyze_result_dict = result.as_dict()

# #             # Define output path: output/docu_intelligence/filename_docs_timestamp.json
# #             output_filename = f"{file_path.stem}_docs_{timestamp}.json"
# #             output_path = OUTPUT_BASE_DIR / output_filename
# #             output_path.parent.mkdir(parents=True, exist_ok=True)

# #             with open(output_path, "w") as f_out:
# #                 json.dump(analyze_result_dict, f_out, indent=4)

# #             print(f"Saved layout data to: {output_path}")

# # if __name__ == "__main__":
# #     process_architecture_diagrams()

# # version 2 with 50% success

# # import os
# # import time
# # import json
# # import base64
# # from pathlib import Path
# # from dotenv import load_dotenv

# # from azure.core.credentials import AzureKeyCredential
# # from azure.ai.documentintelligence import DocumentIntelligenceClient
# # from azure.ai.documentintelligence.models import AnalyzeOutputOption
# # from openai import AzureOpenAI

# # load_dotenv()

# # # Configuration
# # DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
# # DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
# # AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# # AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# # DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# # INPUT_DIR = Path("input")
# # OUTPUT_DIR = Path("output/pipeline_results")
# # INPUT_DIR.mkdir(parents=True, exist_ok=True)
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # def run_document_intelligence(image_path):
# #     print(
# #         f"-> [Step 1] Running Document Intelligence (High-Res) on {image_path.name}..."
# #     )
# #     client = DocumentIntelligenceClient(
# #         endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY)
# #     )

# #     with open(image_path, "rb") as f:
# #         poller = client.begin_analyze_document(
# #             "prebuilt-layout", body=f, output=[AnalyzeOutputOption.FIGURES]
# #         )
# #     return poller.result().as_dict()


# # def minify_spatial_data(raw_doc_json):
# #     print("-> Minifying spatial data...")
# #     page = raw_doc_json.get("pages", [{}])[0]
# #     minified = {"text_blocks": [], "figures": []}

# #     for i, line in enumerate(page.get("lines", [])):
# #         minified["text_blocks"].append(
# #             {
# #                 "id": f"txt_{i}",
# #                 "text": line.get("content"),
# #                 "polygon": line.get("polygon"),
# #             }
# #         )

# #     for j, figure in enumerate(raw_doc_json.get("figures", [])):
# #         regions = figure.get("boundingRegions", [])
# #         if regions:
# #             minified["figures"].append(
# #                 {"id": f"fig_{j}", "polygon": regions[0].get("polygon")}
# #             )
# #     return minified


# # def encode_image(image_path):
# #     with open(image_path, "rb") as image_file:
# #         return base64.b64encode(image_file.read()).decode("utf-8")


# # def generate_architectural_flow_json(image_path, minified_spatial_data):
# #     print("-> [Step 2] Sending data to LLM for Architectural Flow Reconstruction...")
# #     llm_client = AzureOpenAI(
# #         azure_endpoint=AZURE_OPENAI_ENDPOINT,
# #         api_key=AZURE_OPENAI_KEY,
# #         api_version="2024-02-15-preview",
# #     )

# #     base64_image = encode_image(image_path)

# #     prompt = """
# #     You are a Principal Enterprise Cloud Architect. Reconstruct this Azure architecture diagram into a strict, sequential Flow JSON.
# #     DO NOT use graph terminology (no "nodes" or "edges").

# #     You have the image AND a spatial map containing coordinates.

# #     CRITICAL RULES:
# #     1. FIX MULTI-LINE TEXT: If text blocks are vertically stacked with similar X-coordinates (e.g., "Private" directly above "Endpoint", "Azure DNS" above "Private Resolver"), MERGE them into a single component name.
# #     2. ICON MAPPING: Match text labels to nearby 'figure' polygons to confirm it is a physical component.
# #     3. HIERARCHY: Map what lives inside what (Subscription -> VNET -> Subnet -> Component).
# #     4. END-TO-END FLOW: Map the logical traffic sequences step-by-step from source to destination.

# #     OUTPUT SCHEMA (STRICT JSON ONLY):
# #     {
# #       "architectural_hierarchy": [
# #         {
# #           "boundary_name": "Corporate VNET",
# #           "boundary_type": "VNET",
# #           "components": [
# #             {
# #               "name": "Azure DNS Private Resolver",
# #               "type": "DNS Service"
# #             }
# #           ]
# #         }
# #       ],
# #       "end_to_end_flows": [
# #         {
# #           "flow_name": "Ingress User Traffic",
# #           "sequence": [
# #             "User",
# #             "Internet",
# #             "Application Gateway",
# #             "WAF"
# #           ],
# #           "direction": "Inbound",
# #           "traffic_type": "HTTPS"
# #         }
# #       ]
# #     }

# #     Minified Spatial Data:
# #     """ + json.dumps(
# #         minified_spatial_data
# #     )

# #     response = llm_client.chat.completions.create(
# #         model=DEPLOYMENT_NAME,
# #         response_format={"type": "json_object"},
# #         temperature=0.0,
# #         messages=[
# #             {
# #                 "role": "user",
# #                 "content": [
# #                     {"type": "text", "text": prompt},
# #                     {
# #                         "type": "image_url",
# #                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
# #                     },
# #                 ],
# #             }
# #         ],
# #     )
# #     return json.loads(response.choices[0].message.content)


# # def main():
# #     for img_path in INPUT_DIR.glob("*.*"):
# #         if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
# #             continue

# #         print(f"\n========== Processing: {img_path.name} ==========")
# #         timestamp = time.strftime("%Y%m%d-%H%M%S")
# #         base_name = img_path.stem

# #         # Step 1: Extract & Minify
# #         raw_doc_json = run_document_intelligence(img_path)
# #         with open(OUTPUT_DIR / f"{base_name}_step1_raw_{timestamp}.json", "w") as f:
# #             json.dump(raw_doc_json, f, indent=4)

# #         minified_data = minify_spatial_data(raw_doc_json)

# #         # Step 2: LLM Reconstruction (Flow & Structure Only)
# #         final_flow_json = generate_architectural_flow_json(img_path, minified_data)

# #         step2_path = OUTPUT_DIR / f"{base_name}_step2_flow_{timestamp}.json"
# #         with open(step2_path, "w") as f:
# #             json.dump(final_flow_json, f, indent=4)

# #         print(f"========== Success! Step 2 JSON saved to {step2_path} ==========\n")


# # if __name__ == "__main__":
# #     main()


# import os
# import time
# from datetime import datetime
# import json
# import base64
# from pathlib import Path
# from dotenv import load_dotenv

# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from openai import AzureOpenAI

# load_dotenv()

# # Config
# DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
# DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
# AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# INPUT_DIR = Path("input")
# INPUT_DIR.mkdir(parents=True, exist_ok=True)


# def run_document_intelligence(image_path):
#     print(f"-> [Stage 1] Azure OCR & Layout on {image_path.name}...")
#     client = DocumentIntelligenceClient(
#         endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY)
#     )
#     with open(image_path, "rb") as f:
#         poller = client.begin_analyze_document(
#             "prebuilt-layout", body=f, features=["ocrHighResolution"]
#         )

#     print("poller", poller)
#     return poller.result().as_dict()


# def minify_spatial_data(raw_doc_json):
#     minified = {"spatial_text": [], "boundaries": []}

#     for i, para in enumerate(raw_doc_json.get("paragraphs", [])):
#         content = para.get("content")
#         minified["spatial_text"].append(
#             {
#                 "id": f"p_{i}",
#                 "text": content,
#                 "polygon": (
#                     para.get("boundingRegions")[0].get("polygon")
#                     if para.get("boundingRegions")
#                     else []
#                 ),
#             }
#         )

#     for j, figure in enumerate(raw_doc_json.get("figures", [])):
#         regions = figure.get("boundingRegions", [])
#         if regions:
#             minified["boundaries"].append(
#                 {"id": f"fig_{j}", "polygon": regions[0].get("polygon")}
#             )

#     print("minified", minified)
#     return minified


# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# def extract_connections(image_path):
#     print("-> [Stage 2] GPT-4o Vision: Analyzing Legend & Extracting Connectivity...")
#     llm_client = AzureOpenAI(
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_KEY,
#         api_version=AZURE_OPENAI_VERSION,
#     )

#     prompt = """
#         Analyze this technical architecture diagram. You are a precise visual routing expert.

#         STEP 1: AUTONOMOUS LEGEND & SEMANTIC ANALYSIS
#         Before tracing any lines, scan the entire image for a Legend or consistent visual patterns.
#         - Autonomously determine what different colors, line thicknesses, and styles (e.g., dashed vs. solid) represent in the context of this specific image.
#         - Look for sequencing markers (numbered or lettered steps) that dictate traffic flow.

#         STEP 2: STRICT HOP-BY-HOP EXTRACTION
#         Extract the connections by following these absolute geometric rules:
#         1. DO NOT SKIP INTERMEDIARIES: If a line passes through, touches, or stops at an icon or component boundary, that component is a MANDATORY hop. You must break the path into segments (e.g., Source -> Intermediary Node, Intermediary Node -> Destination).
#         2. FIND THE ARROWHEAD: To determine the 'target' and direction, you MUST physically locate the arrowhead drawn on the line. Do not guess direction based on component names or logical assumptions. 
#         3. INTERSECTIONS ARE NOT HOPS: If two lines merely cross over each other without a distinct connecting node or dot, they are independent lines. Do not map them as connected.

#         OUTPUT FORMAT (STRICT JSON):
#         {
#         "connections": [
#             {
#             "source": "[Exact Name of Source Component]",
#             "target": "[Exact Name of Target Component]",
#             "flow": "One-way | Bi-directional | Unknown",
#             "style_and_meaning": "[Color/Style] ([Inferred Meaning from Legend])",
#             "sequence": "[Step Number/Letter or 'None']"
#             }
#         ]
#         }
#     """

#     response = llm_client.chat.completions.create(
#         model=DEPLOYMENT_NAME,
#         response_format={"type": "json_object"},
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
#                         },
#                     },
#                 ],
#             }
#         ],
#     )
#     return json.loads(response.choices[0].message.content)


# def consolidate_architecture(image_path, spatial_data, connections):
#     print("-> [Stage 3] Consolidating into Final Architectural JSON...")
#     llm_client = AzureOpenAI(
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_KEY,
#         api_version=AZURE_OPENAI_VERSION,
#     )

#     prompt = f"""
#         You are a Principal Cloud Architect and Spatial Data Synthesizer. Synthesize the provided Spatial Map and Connection List into a single, accurate JSON representation.
        
#         INPUTS:
#         1. Spatial Map (OCR Text + X/Y Polygons): {json.dumps(spatial_data)}
#         2. Connection List (Legend-aware geometric relationships): {json.dumps(connections)}
        
#         CRITICAL ALGORITHMIC RULES:
#         1. INFER BOUNDARIES FROM PROXIMITY (THE FALLBACK RULE): If explicit boundary polygons are missing from the spatial map, you MUST use the X/Y polygon coordinates of the text to infer boundaries. If a cluster of services is located directly beneath a header text or grouped closely together, nest them under that header.
#         2. NO DEDUPLICATION (COORDINATE NAMESPACING): Diagrams often have identical components in different physical zones. If you see identical text labels with significantly different X/Y coordinates, DO NOT merge them. Treat them as distinct instances. Differentiate them in your output using their spatial context (e.g., "Component Name (Left Zone)").
#         3. SEPARATE FLOW FROM STRUCTURE: Do NOT nest components inside one another simply because a flow arrow connects them. Hierarchy is dictated strictly by physical enclosure or tight spatial clustering.
#         4. ZERO DROP POLICY & EXTERNAL ENTITIES: Every single text block from the Spatial Map MUST be accounted for. If a component (e.g., a user icon, branch office, or floating monitoring tool) sits physically outside the main structural clusters, do not force it into the hierarchy or delete it. Place it in the "external_and_unbound_entities" array.
        
#         OUTPUT FORMAT (STRICT JSON):
#         {{
#         "architectural_hierarchy": [
#             {{
#             "name": "[Main Boundary/Region Name]",
#             "type": "[Boundary Type]",
#             "children": [
#                 {{
#                 "name": "[Nested Boundary/VPC/VNET Name]",
#                 "type": "[Boundary Type]",
#                 "children": ["[Component 1]", "[Component 2]"]
#                 }}
#             ]
#             }}
#         ],
#         "external_and_unbound_entities": [
#             "[Floating Entity 1]",
#             "[External Network 2]"
#         ],
#         "end_to_end_flows": [...]
#         }}
#     """

#     response = llm_client.chat.completions.create(
#         model=DEPLOYMENT_NAME,
#         response_format={"type": "json_object"},
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
#                         },
#                     },
#                 ],
#             }
#         ],
#     )
#     return json.loads(response.choices[0].message.content)


# def main():
#     date_str = datetime.now().strftime("%Y-%m-%d")
#     output_dir = Path(f"output/pipeline_results/{date_str}")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for img_path in INPUT_DIR.glob("*.*"):
#         if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
#             continue

#         print(f"\n========== Processing: {img_path.name} ==========")
#         timestamp = time.strftime("%H%M%S")
#         base_name = img_path.stem

#         # Stage 1: Document Intelligence
#         raw_ocr = run_document_intelligence(img_path)
#         minified_ocr = minify_spatial_data(raw_ocr)

#         step1_path = output_dir / f"{base_name}_step1_spatial_{timestamp}.json"
#         with open(step1_path, "w") as f:
#             json.dump(minified_ocr, f, indent=4)

#         # Stage 2: Visual Connection Mapping
#         connections = extract_connections(img_path)

#         step2_path = output_dir / f"{base_name}_step2_connections_{timestamp}.json"
#         with open(step2_path, "w") as f:
#             json.dump(connections, f, indent=4)

#         # Stage 3: Logical Consolidation
#         final_json = consolidate_architecture(img_path, minified_ocr, connections)

#         step3_path = output_dir / f"{base_name}_step3_final_{timestamp}.json"
#         with open(step3_path, "w") as f:
#             json.dump(final_json, f, indent=4)

#         print(f"-> SUCCESS: Files saved to {output_dir}")


# if __name__ == "__main__":
#     main()

import os
import time
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption
from openai import AzureOpenAI

load_dotenv()

# Configuration
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

INPUT_DIR = Path("input")
# Ensure output matches your preferred timestamped directory structure
date_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = Path(f"output/pipeline_results/{date_str}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_stage1_doc_intelligence(image_path):
    print(f"-> [Stage 1] Azure OCR & Layout (High-Res) on {image_path.name}...")
    client = DocumentIntelligenceClient(DOC_INTEL_ENDPOINT, AzureKeyCredential(DOC_INTEL_KEY))
    
    with open(image_path, "rb") as f:
        # We explicitly request FIGURES to find icon locations
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=f, 
            features=["ocrHighResolution"],
            output=[AnalyzeOutputOption.FIGURES]
        )
    result = poller.result().as_dict()
    return result

def run_stage2_cv_geometry(image_path):
    print(f"-> [Stage 2] OpenCV: Detecting Lines (Hough) and Arrowheads...")
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect Lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=15)
    
    line_data = []
    if lines is not None:
        for l in lines:
            line_data.append(l[0].tolist()) # [x1, y1, x2, y2]

    # Detect Arrowheads using Contour Analysis (looking for triangles)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    arrowheads = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3: # Triangle
            # Store centroid of the triangle
            M = cv2.moments(approx)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                arrowheads.append({"point": [cx, cy], "polygon": approx.tolist()})

    return {"lines": line_data, "arrowheads": arrowheads}

def minify_and_merge(raw_ocr, geometry):
    print("-> [Stage 3] Merging OCR with Geometry into Minified Spatial Map...")
    minified = {
        "spatial_text": [],
        "icons": [],
        "detected_connectors": geometry["lines"],
        "detected_arrowheads": geometry["arrowheads"]
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

def run_stage4_consolidation(image_path, minified_data):
    print("-> [Stage 4] GPT-4o: Consolidating Hierarchy & Semantic Flow...")
    llm_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION
    )

    prompt = f"""
    You are a Principal Enterprise Architect. Reconstruct the provided Azure Architecture Diagram into a single Consolidated JSON.
    
    INPUT DATA:
    - Minified Spatial Map (Text, Icons, Lines, Arrows): {json.dumps(minified_data)}
    
    YOUR TASK:
    1. HIERARCHY: Identify nested boundaries (Subscriptions, VNETs, Subnets). Use the X/Y polygons to see which components sit inside which boundaries.
    2. ICON IDENTIFICATION: Cross-reference the 'icons' coordinates with the image to identify the specific Azure service (e.g., 'f_1' is a 'Load Balancer').
    3. CONNECTIVITY: Use the 'detected_connectors' (Hough lines) and 'detected_arrowheads' to trace traffic flow. 
       - If a line connects Component A to Component B and has an arrowhead near B, direction is A -> B.
    4. NO DEDUPLICATION: If two components have the same name but different coordinates, treat them as separate instances.

    OUTPUT SCHEMA (STRICT JSON):
    {{
        "infrastructure_hierarchy": [
            {{ "name": "boundary_name", "type": "vnet/subnet", "components": [] }}
        ],
        "logical_connectivity": [
            {{ "source": "", "target": "", "direction": "", "purpose": "" }}
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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                ]
            }
        ]
    )
    return json.loads(response.choices[0].message.content)

def main():
    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue

        print(f"\n========== STARTING PIPELINE: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        
        # Step 1: Azure Doc Intelligence
        raw_ocr = run_stage1_doc_intelligence(img_path)
        with open(OUTPUT_DIR / f"{img_path.stem}_step1_raw_{timestamp}.json", "w") as f:
            json.dump(raw_ocr, f, indent=4)

        # Step 2: OpenCV Geometry
        geometry = run_stage2_cv_geometry(img_path)
        with open(OUTPUT_DIR / f"{img_path.stem}_step2_geometry_{timestamp}.json", "w") as f:
            json.dump(geometry, f, indent=4)

        # Step 3: Minification
        minified_data = minify_and_merge(raw_ocr, geometry)
        with open(OUTPUT_DIR / f"{img_path.stem}_step3_spatial_{timestamp}.json", "w") as f:
            json.dump(minified_data, f, indent=4)

        # Step 4: LLM Consolidation
        final_json = run_stage4_consolidation(img_path, minified_data)
        with open(OUTPUT_DIR / f"{img_path.stem}_step4_final_{timestamp}.json", "w") as f:
            json.dump(final_json, f, indent=4)

        print(f"========== SUCCESS: {img_path.name} COMPLETE ==========\n")

if __name__ == "__main__":
    main()