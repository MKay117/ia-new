# import os
# import time
# import json
# import base64
# import mimetypes
# import requests
# from dotenv import load_dotenv
# from openai import AzureOpenAI

# # Load credentials
# load_dotenv()
# AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
# AZURE_KEY = os.getenv("AZURE_VISION_KEY")

# # Azure Open AI credentials
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# # Initialize OpenAI Client
# llm_client = AzureOpenAI(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
#     api_version=AZURE_OPENAI_VERSION,
# )

# # Ensure directories exist
# os.makedirs("input", exist_ok=True)
# os.makedirs("output", exist_ok=True)


# def get_mime_type(file_path):
#     """Step 1: Determine MIME type."""
#     mime_type, _ = mimetypes.guess_type(file_path)
#     return mime_type or "application/octet-stream"


# def run_azure_ocr(image_path):
#     """Step 2: Fetch text, polygons, and metadata via Azure."""
#     url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
#     headers = {
#         "Ocp-Apim-Subscription-Key": AZURE_KEY,
#         "Content-Type": "application/octet-stream",
#     }

#     with open(image_path, "rb") as image_file:
#         image_data = image_file.read()

#     response = requests.post(url, headers=headers, data=image_data)
#     response.raise_for_status()
#     operation_url = response.headers["Operation-Location"]

#     # Poll for results
#     while True:
#         poll_response = requests.get(
#             operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY}
#         )
#         result = poll_response.json()
#         if result["status"] not in ["notStarted", "running"]:
#             break
#         time.sleep(1)

#     return result


# def minify_ocr_data(raw_azure_data):
#     """Minifies raw Azure JSON into a lightweight payload."""
#     minified_payload = {"metadata": {}, "lines": []}

#     if raw_azure_data["status"] == "succeeded":
#         read_result = raw_azure_data["analyzeResult"]["readResults"][0]

#         # Image Metadata
#         minified_payload["metadata"] = {
#             "width": read_result.get("width"),
#             "height": read_result.get("height"),
#             "angle": read_result.get("angle"),
#         }

#         # Text, Polygons, and Confidence
#         for line in read_result.get("lines", []):
#             words = line.get("words", [])
#             confidences = [w.get("confidence", 1.0) for w in words if "confidence" in w]
#             avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

#             minified_payload["lines"].append(
#                 {
#                     "text": line["text"],
#                     "polygon": line["boundingBox"],
#                     "confidence": round(avg_confidence, 4),
#                 }
#             )

#     return minified_payload


# def encode_image_base64(image_path):
#     """Encodes image for the LLM."""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# def run_llm_correction(image_path, mime_type, minified_data):
#     """Step 4: Send Image and Minified Data to GPT-4o."""
#     base64_image = encode_image_base64(image_path)

#     # Explicitly asking for JSON is required by OpenAI's API when using json_object mode
#     prompt = f"""
#     You are an expert OCR correction assistant. 
#     I am providing you with an architecture diagram and the raw OCR data extracted from it.

#     The raw OCR data often incorrectly separates multi-line labels (e.g., splitting "Microsoft" and "Windows" into two items even though they form one visual label). 
#     Your task is to:
#     1. Look at the image to understand the visual layout of the text.
#     2. Use the provided OCR bounding box data to accurately group text that visually belongs together as a single label, title, or block of text. 
#     3. If text is floating or inside a boundary box, group it logically based on its visual proximity and alignment.
#     4. Return a JSON object with a single key 'text_blocks', containing an array of objects. Each object must have 'text' (the corrected, combined string), 'confidence' (averaged if combined), and 'polygon' (a new bounding box that encompasses all combined text).

#     Raw OCR Data:
#     {json.dumps(minified_data)}
#     """

#     response = llm_client.chat.completions.create(
#         model="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
#         temperature=0.0,
#         max_tokens=5000,
#         response_format={"type": "json_object"},
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
#                     },
#                 ],
#             }
#         ],
#     )

#     return json.loads(response.choices[0].message.content)


# def main():
#     for filename in os.listdir("input"):
#         file_path = os.path.join("input", filename)

#         if not os.path.isfile(file_path):
#             continue

#         print(f"\n--- Processing: {filename} ---")
#         base_name = os.path.splitext(filename)[0]
#         timestamp = time.strftime("%Y%m%d-%H%M%S")

#         # Step 1: MIME Type
#         mime_type = get_mime_type(file_path)
#         print(f"Detected MIME type: {mime_type}")

#         # Step 2: Azure OCR
#         print("Running Azure OCR (Stage 1)...")
#         stage_1_output = run_azure_ocr(file_path)

#         # Step 3: Save Stage 1 Output
#         stage_1_filename = f"output/{base_name}_stage_1_{timestamp}.json"
#         with open(stage_1_filename, "w") as f:
#             json.dump(stage_1_output, f, indent=4)
#         print(f"Saved raw OCR to: {stage_1_filename}")

#         # Minify for LLM
#         minified_data = minify_ocr_data(stage_1_output)

#         print("\n--- Minified JSON Payload for LLM ---")
#         print(json.dumps(minified_data, indent=2))
#         print("-------------------------------------\n")

#         # Step 4: LLM Correction
#         print("Running GPT-4o Correction (Stage 2)...")
#         try:
#             stage_2_output = run_llm_correction(file_path, mime_type, minified_data)

#             # Step 5: Save Stage 2 Output
#             stage_2_filename = f"output/{base_name}_stage_2_{timestamp}.json"
#             with open(stage_2_filename, "w") as f:
#                 json.dump(stage_2_output, f, indent=4)
#             print(f"Saved LLM Output to: {stage_2_filename}")

#         except Exception as e:
#             print(f"LLM Processing failed for {filename}: {e}")


# if __name__ == "__main__":
#     main()



# Above code is Version 1 in which stage 1 is perfect whereas stage 2 doesnt work. Below is Version 2 
# Stage 1 output is given an ID and LLM call only suggest which IDs can be grouped together

import os
import time
import json
import base64
import mimetypes
import requests
import traceback
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load credentials
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")

# Azure Open AI credentials
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Initialize OpenAI Client
llm_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_VERSION,
)

# Ensure directories exist
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


def get_mime_type(file_path):
    """Step 1: Determine MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def run_azure_ocr(image_path):
    """Step 2: Fetch text, polygons, and metadata via Azure."""
    url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream",
    }

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = requests.post(url, headers=headers, data=image_data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]

    # Poll for results
    while True:
        poll_response = requests.get(
            operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY}
        )
        result = poll_response.json()
        if result["status"] not in ["notStarted", "running"]:
            break
        time.sleep(1)

    return result


def minify_and_map_ocr_data(raw_azure_data):
    """Assigns unique IDs to Azure OCR output for the Failsafe architecture."""
    minified_payload = {"metadata": {}, "items": []}

    if raw_azure_data["status"] == "succeeded":
        read_result = raw_azure_data["analyzeResult"]["readResults"][0]

        # Image Metadata
        minified_payload["metadata"] = {
            "width": read_result.get("width"),
            "height": read_result.get("height"),
            "angle": read_result.get("angle"),
        }

        # Text, Polygons, Confidence, and unique ID mapping
        item_id = 1
        for line in read_result.get("lines", []):
            words = line.get("words", [])
            confidences = [w.get("confidence", 1.0) for w in words if "confidence" in w]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

            minified_payload["items"].append(
                {
                    "id": item_id,
                    "text": line["text"],
                    "polygon": line["boundingBox"],
                    "confidence": round(avg_confidence, 4),
                }
            )
            item_id += 1

    return minified_payload


def encode_image_base64(image_path):
    """Encodes image for the LLM."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def run_llm_grouping(image_path, mime_type, minified_data):
    """Step 4: Execute Typology-Aware Grouping and Domain Correction."""
    base64_image = encode_image_base64(image_path)

    llm_payload = [{"id": item["id"], "text": item["text"], "polygon": item["polygon"]} for item in minified_data["items"]]

    prompt = f"""
    You are a Principal Enterprise Architect and Computer Vision post-processor operating in a BFSI environment.
    Your task is to take raw, disjointed OCR bounding boxes from a cloud architecture diagram and logically group them into coherent semantic entities.

    CRITICAL VISUAL GRAMMAR RULES FOR ARCHITECTURE DIAGRAMS:

    1. DOMAIN VOCABULARY CORRECTION (Strict):
       - Correct common OCR typos. Change "Al" to "AI" (e.g., "Al Foundry" -> "AI Foundry").
       - Fix casing: "DDos" -> "DDoS", "Vnet" -> "VNet", "Entra ld" -> "Entra ID".

    2. MULTI-LINE COMPONENT LABELS (Merge):
       - Group text that forms a single logical node broken across multiple lines due to tight spatial stacking (e.g., "Azure DNS" directly above "Private Resolver").
       
    3. BOUNDARIES & CONTAINER TITLES (Do NOT Merge with Contents):
       - A title at the top, bottom, or corner of a boundary box, subnet, VPC, or trust zone (e.g., "AI Services Virtual Network", "Corporate VNET") must NEVER be merged with the nodes/icons floating inside that box. 

    4. FLOW AND ARROW LABELS (Isolate):
       - Text floating next to lines, arrows, or transit paths (e.g., "Feedback Loop", "Replication", "HTTPS", "Port 443", "JSON") are atomic flow descriptors. DO NOT merge them with the source or destination boxes.

    5. SEQUENCE MARKERS (Isolate):
       - Floating numbers or letters inside circles/badges (e.g., "1", "2", "a", "b", "Step 1") represent operational stages. Isolate them.

    6. NETWORK METADATA (Isolate):
       - Standalone IP addresses, CIDR blocks (e.g., "10.0.0.0/24"), or ports (e.g., ":8080") must be isolated as metadata.

    OUTPUT SCHEMA:
    Provide a JSON object grouping the IDs, providing the corrected string, and classifying the architectural type.
    Valid types are: "Component_Node", "Boundary_Title", "Flow_Label", "Stage_Marker", "Network_Metadata", "Unknown".
    
    {{
      "blocks": [
        {{"ids": [1, 2], "label": "Azure DNS Private Resolver", "type": "Component_Node"}},
        {{"ids": [15, 16], "label": "AI Services Virtual Network", "type": "Boundary_Title"}},
        {{"ids": [22], "label": "HTTPS", "type": "Flow_Label"}},
        {{"ids": [5], "label": "1", "type": "Stage_Marker"}}
      ]
    }}

    OCR Data:
    {json.dumps(llm_payload)}
    """

    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.0, 
        max_tokens=4000,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


def merge_polygons(polygons):
    """Calculates a new outer bounding box that encompasses all provided 8-point polygons."""
    xs = [x for poly in polygons for x in poly[0::2]]
    ys = [y for poly in polygons for y in poly[1::2]]
    
    # If no valid coordinates, return a default empty box
    if not xs or not ys:
        return [0, 0, 0, 0, 0, 0, 0, 0]
        
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Return standard 8-point polygon: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    return [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]


def reconstruct_text_blocks(minified_data, llm_data):
    """Step 5: Reassemble data with Failsafe mapping and new Typology fields."""
    final_blocks = []
    processed_ids = set()
    
    item_lookup = {item["id"]: item for item in minified_data["items"]}
    
    # 1. Process the LLM's typologically sorted blocks
    for block in llm_data.get("blocks", []):
        group_ids = block.get("ids", [])
        corrected_label = block.get("label", "")
        block_type = block.get("type", "Unknown")
        
        if not group_ids or not isinstance(group_ids, list):
            continue
            
        group_items = []
        for item_id in group_ids:
            if item_id in item_lookup and item_id not in processed_ids:
                group_items.append(item_lookup[item_id])
                processed_ids.add(item_id)
        
        if not group_items:
            continue
            
        merged_polygon = merge_polygons([item["polygon"] for item in group_items])
        avg_confidence = sum([item["confidence"] for item in group_items]) / len(group_items)
        
        final_blocks.append({
            "text": corrected_label,
            "type": block_type,
            "confidence": round(avg_confidence, 4),
            "polygon": merged_polygon,
            "merged_ids": group_ids
        })
        
    # 2. FAILSAFE: Catch anything the LLM skipped to prevent data loss
    for item in minified_data["items"]:
        if item["id"] not in processed_ids:
            # Programmatic fallback for the AI typo on skipped items
            safe_text = item["text"].replace("Al ", "AI ").replace(" Al", " AI")
            if item["text"] == "Al": safe_text = "AI"
            
            final_blocks.append({
                "text": safe_text,
                "type": "Ungrouped_Raw",
                "confidence": item["confidence"],
                "polygon": item["polygon"],
                "merged_ids": [item["id"]]
            })
            
    # Sort top to bottom, left to right
    final_blocks.sort(key=lambda block: (block["polygon"][1], block["polygon"][0]))
    
    return {"text_blocks": final_blocks}


def main():
    for filename in os.listdir("input"):
        file_path = os.path.join("input", filename)

        if not os.path.isfile(file_path):
            continue

        print(f"\n--- Processing: {filename} ---")
        base_name = os.path.splitext(filename)[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        mime_type = get_mime_type(file_path)

        # Step 1: Azure OCR
        print("1. Running Azure OCR (Stage 1)...")
        stage_1_output = run_azure_ocr(file_path)

        stage_1_filename = f"output/{base_name}_stage_1_{timestamp}.json"
        with open(stage_1_filename, "w") as f:
            json.dump(stage_1_output, f, indent=4)
        print(f"   Saved raw OCR to: {stage_1_filename}")

        # Map IDs and Minify
        minified_data = minify_and_map_ocr_data(stage_1_output)

        # Step 2: LLM Grouping
        print("2. Running GPT-4o Grouping Decision...")
        try:
            llm_groups = run_llm_grouping(file_path, mime_type, minified_data)
            # BUG FIX: Ensure we print the 'blocks' key correctly
            print(f"   LLM suggested groupings: {llm_groups.get('blocks', [])}")
            
            # Step 3: Failsafe Assembly
            print("3. Executing Failsafe Assembly (Zero Data Loss)...")
            stage_2_output = reconstruct_text_blocks(minified_data, llm_groups)

            stage_2_filename = f"output/{base_name}_stage_2_{timestamp}.json"
            with open(stage_2_filename, "w") as f:
                json.dump(stage_2_output, f, indent=4)
            print(f"   Saved LLM Output to: {stage_2_filename}")

        except Exception as e:
            print(f"   Processing failed for {filename}: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()



