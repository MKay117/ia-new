# import os
# import json
# import time
# from datetime import datetime
# from pathlib import Path
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeResult

# # Configuration [cite: 27, 51]
# ENDPOINT = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
# KEY = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
# INPUT_DIR = Path("input")
# OUTPUT_BASE_DIR = Path("output/docu_intelligence")

# def process_architecture_diagrams():
#     # Instantiate client [cite: 27]
#     client = DocumentIntelligenceClient(ENDPOINT, AzureKeyCredential(KEY))
    
#     # Ensure output directory exists
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
#     if not INPUT_DIR.exists():
#         print(f"Input directory {INPUT_DIR} not found.")
#         return

#     for file_path in INPUT_DIR.iterdir():
#         if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf', '.tiff']:
#             print(f"Processing: {file_path.name}...")
            
#             with open(file_path, "rb") as f:
#                 # Step 1: Extract Layout (Text, Tables, Selection Marks, Polygons) [cite: 51, 52]
#                 poller = client.begin_analyze_document("prebuilt-layout", body=f)
#                 result: AnalyzeResult = poller.result() # Wait for long-running operation 

#             # Convert to JSON-serializable dictionary 
#             analyze_result_dict = result.as_dict()

#             # Define output path: output/docu_intelligence/filename_docs_timestamp.json
#             output_filename = f"{file_path.stem}_docs_{timestamp}.json"
#             output_path = OUTPUT_BASE_DIR / output_filename
#             output_path.parent.mkdir(parents=True, exist_ok=True)

#             with open(output_path, "w") as f_out:
#                 json.dump(analyze_result_dict, f_out, indent=4)
            
#             print(f"Saved layout data to: {output_path}")

# if __name__ == "__main__":
#     process_architecture_diagrams()

# import os
# import time
# import json
# import base64
# from pathlib import Path
# from dotenv import load_dotenv

# from azure.core.credentials import AzureKeyCredential
# from azure.ai.documentintelligence import DocumentIntelligenceClient
# from azure.ai.documentintelligence.models import AnalyzeOutputOption
# from openai import AzureOpenAI

# load_dotenv()

# # Configuration
# DOC_INTEL_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
# DOC_INTEL_KEY = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# INPUT_DIR = Path("input")
# OUTPUT_DIR = Path("output/pipeline_results")
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def run_document_intelligence(image_path):
#     print(f"-> [Step 1] Running Document Intelligence (High-Res) on {image_path.name}...")
#     client = DocumentIntelligenceClient(endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY))
    
#     with open(image_path, "rb") as f:
#         poller = client.begin_analyze_document(
#             "prebuilt-layout", 
#             body=f,
#             output=[AnalyzeOutputOption.FIGURES]
#         )
#     return poller.result().as_dict()

# def minify_spatial_data(raw_doc_json):
#     print("-> Minifying spatial data...")
#     page = raw_doc_json.get("pages", [{}])[0]
#     minified = {"text_blocks": [], "figures": []}

#     for i, line in enumerate(page.get("lines", [])):
#         minified["text_blocks"].append({
#             "id": f"txt_{i}",
#             "text": line.get("content"),
#             "polygon": line.get("polygon")
#         })
        
#     for j, figure in enumerate(raw_doc_json.get("figures", [])):
#         regions = figure.get("boundingRegions", [])
#         if regions:
#             minified["figures"].append({
#                 "id": f"fig_{j}",
#                 "polygon": regions[0].get("polygon")
#             })
#     return minified

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def generate_architectural_flow_json(image_path, minified_spatial_data):
#     print("-> [Step 2] Sending data to LLM for Architectural Flow Reconstruction...")
#     llm_client = AzureOpenAI(
#         azure_endpoint=AZURE_OPENAI_ENDPOINT,
#         api_key=AZURE_OPENAI_KEY,
#         api_version="2024-02-15-preview"
#     )
    
#     base64_image = encode_image(image_path)
    
#     prompt = """
#     You are a Principal Enterprise Cloud Architect. Reconstruct this Azure architecture diagram into a strict, sequential Flow JSON.
#     DO NOT use graph terminology (no "nodes" or "edges").
    
#     You have the image AND a spatial map containing coordinates.
    
#     CRITICAL RULES:
#     1. FIX MULTI-LINE TEXT: If text blocks are vertically stacked with similar X-coordinates (e.g., "Private" directly above "Endpoint", "Azure DNS" above "Private Resolver"), MERGE them into a single component name.
#     2. ICON MAPPING: Match text labels to nearby 'figure' polygons to confirm it is a physical component.
#     3. HIERARCHY: Map what lives inside what (Subscription -> VNET -> Subnet -> Component).
#     4. END-TO-END FLOW: Map the logical traffic sequences step-by-step from source to destination.
    
#     OUTPUT SCHEMA (STRICT JSON ONLY):
#     {
#       "architectural_hierarchy": [
#         {
#           "boundary_name": "Corporate VNET",
#           "boundary_type": "VNET",
#           "components": [
#             {
#               "name": "Azure DNS Private Resolver",
#               "type": "DNS Service"
#             }
#           ]
#         }
#       ],
#       "end_to_end_flows": [
#         {
#           "flow_name": "Ingress User Traffic",
#           "sequence": [
#             "User",
#             "Internet",
#             "Application Gateway",
#             "WAF"
#           ],
#           "direction": "Inbound",
#           "traffic_type": "HTTPS"
#         }
#       ]
#     }
    
#     Minified Spatial Data:
#     """ + json.dumps(minified_spatial_data)

#     response = llm_client.chat.completions.create(
#         model=DEPLOYMENT_NAME,
#         response_format={"type": "json_object"},
#         temperature=0.0,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
#                 ]
#             }
#         ]
#     )
#     return json.loads(response.choices[0].message.content)

# def main():
#     for img_path in INPUT_DIR.glob("*.*"):
#         if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
#             continue
            
#         print(f"\n========== Processing: {img_path.name} ==========")
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         base_name = img_path.stem
        
#         # Step 1: Extract & Minify
#         raw_doc_json = run_document_intelligence(img_path)
#         with open(OUTPUT_DIR / f"{base_name}_step1_raw_{timestamp}.json", "w") as f:
#             json.dump(raw_doc_json, f, indent=4)
            
#         minified_data = minify_spatial_data(raw_doc_json)
        
#         # Step 2: LLM Reconstruction (Flow & Structure Only)
#         final_flow_json = generate_architectural_flow_json(img_path, minified_data)
        
#         step2_path = OUTPUT_DIR / f"{base_name}_step2_flow_{timestamp}.json"
#         with open(step2_path, "w") as f:
#             json.dump(final_flow_json, f, indent=4)
            
#         print(f"========== Success! Step 2 JSON saved to {step2_path} ==========\n")

# if __name__ == "__main__":
#     main()
import os
import time
from datetime import datetime
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.openai import AzureOpenAI

load_dotenv()

# Config
DOC_INTEL_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

INPUT_DIR = Path("input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_document_intelligence(image_path):
    print(f"-> [Stage 1] Azure OCR & Layout on {image_path.name}...")
    client = DocumentIntelligenceClient(endpoint=DOC_INTEL_ENDPOINT, credential=AzureKeyCredential(DOC_INTEL_KEY))
    with open(image_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            body=f,
            features=["ocrHighResolution"] 
        )
    return poller.result().as_dict()

def minify_spatial_data(raw_doc_json):
    minified = {"spatial_text": [], "boundaries": []}
    
    for i, para in enumerate(raw_doc_json.get("paragraphs", [])):
        content = para.get("content")
        minified["spatial_text"].append({
            "id": f"p_{i}",
            "text": content,
            "polygon": para.get("boundingRegions")[0].get("polygon") if para.get("boundingRegions") else []
        })
        
    for j, figure in enumerate(raw_doc_json.get("figures", [])):
        regions = figure.get("boundingRegions", [])
        if regions:
            minified["boundaries"].append({
                "id": f"fig_{j}",
                "polygon": regions[0].get("polygon")
            })
    return minified

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_connections(image_path):
    print("-> [Stage 2] GPT-4o Vision: Analyzing Legend & Extracting Connectivity...")
    llm_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2024-02-15-preview")
    
    prompt = """
    Analyze this technical architecture diagram. You are a networking routing expert.

    STEP 1: VISUAL LEGEND & SEMANTICS
    Before tracing any lines, scan the entire image for a Legend or implicit visual semantics. 
    - Note what different colors mean (e.g., Orange = SD-WAN, Blue = GRE, Green = Ingress).
    - Note what line styles mean (e.g., Dashed = Peering/Logical, Solid = Physical/Direct).
    - Look for sequencing markers (e.g., numbered steps like 1, 2, 3 or lettered steps like A, B, C).

    STEP 2: EXTRACT CONNECTIONS
    Focus ONLY on the lines and arrows connecting components. Using the context from the legend, extract:
    1. The source component name and the destination component name.
    2. The direction (One-way, Bi-directional).
    3. The semantic style (Incorporate the color, line style, and meaning based on the legend).
    4. The sequence step (if labelled on the line).

    OUTPUT SCHEMA (STRICT JSON):
    {
      "connections": [
        {
          "source": "Internet",
          "target": "WAF",
          "flow": "One-way",
          "style_and_meaning": "Solid Green (Ingress Traffic)",
          "sequence": "None"
        },
        {
          "source": "Virtual Private Gateway",
          "target": "TGW ENI",
          "flow": "One-way",
          "style_and_meaning": "Solid Orange (SD-WAN overlay)",
          "sequence": "3"
        }
      ]
    }
    """
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
        ]}]
    )
    return json.loads(response.choices[0].message.content)

def consolidate_architecture(image_path, spatial_data, connections):
    print("-> [Stage 3] Consolidating into Final Architectural JSON...")
    llm_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version="2024-02-15-preview")
    
    prompt = f"""
    You are a Principal Cloud Architect. Synthesize the provided Spatial Map and Connection List into a single source of truth.
    
    INPUTS:
    1. Spatial Map (OCR Text + Positions): {json.dumps(spatial_data)}
    2. Connection List (Legend-aware geometric relationships): {json.dumps(connections)}
    
    RULES:
    - Use the Spatial Map to define the Hierarchy (what is inside what).
    - Use the Connection List to define the end_to_end_flows. Ensure you retain the specific semantic styles and sequence steps.
    - If a connection target name in the List slightly differs from the OCR text, prioritize the OCR text to link them properly.
    - DO NOT drop components from the hierarchy just because they lack a hard bounding box (e.g., Gateways on borders, or external services). Place them logically.
    
    SCHEMA:
    {{
      "architectural_hierarchy": [...],
      "end_to_end_flows": [...]
    }}
    """
    
    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
        ]}]
    )
    return json.loads(response.choices[0].message.content)

def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f"output/pipeline_results/{date_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}: continue
            
        print(f"\n========== Processing: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem
        
        # Stage 1: Document Intelligence
        raw_ocr = run_document_intelligence(img_path)
        minified_ocr = minify_spatial_data(raw_ocr)
        
        step1_path = output_dir / f"{base_name}_step1_spatial_{timestamp}.json"
        with open(step1_path, "w") as f:
            json.dump(minified_ocr, f, indent=4)
            
        # Stage 2: Visual Connection Mapping
        connections = extract_connections(img_path)
        
        step2_path = output_dir / f"{base_name}_step2_connections_{timestamp}.json"
        with open(step2_path, "w") as f:
            json.dump(connections, f, indent=4)
            
        # Stage 3: Logical Consolidation
        final_json = consolidate_architecture(img_path, minified_ocr, connections)
        
        step3_path = output_dir / f"{base_name}_step3_final_{timestamp}.json"
        with open(step3_path, "w") as f:
            json.dump(final_json, f, indent=4)
            
        print(f"-> SUCCESS: Files saved to {output_dir}")

if __name__ == "__main__":
    main()
