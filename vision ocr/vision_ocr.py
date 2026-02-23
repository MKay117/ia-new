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
    """Step 4: Ask LLM ONLY for the grouping logic (IDs), not the text generation."""
    base64_image = encode_image_base64(image_path)

    # Strip the payload to save tokens; the LLM only needs ID, Text, and Polygon
    llm_payload = [{"id": item["id"], "text": item["text"], "polygon": item["polygon"]} for item in minified_data["items"]]

    prompt = f"""
    You are an expert cloud architecture diagram OCR post-processor.
    I am providing you with an image of an architecture diagram and its extracted OCR text. 
    Each text line has a unique integer "id", the "text" itself, and a "polygon" (bounding box coordinates).

    Your ONLY job is to identify which IDs should be visually merged together into a single text label, and output an array of those grouped IDs.

    CRITICAL ANTI-OVER-GROUPING RULES:
    1. ONLY group text if it forms a SINGLE logical label broken across multiple lines (e.g., "Azure" directly above "Firewall").
    2. DO NOT group distinct nodes or components just because they are inside the same boundary box, container, or network zone. 
    3. DO NOT group separate steps in a flowchart or flow diagram. Keep them separate.
    4. DO NOT group a container's title with the items inside it. 
    5. DO NOT group unrelated standalone words (like "User" and "Internet").
    6. DO NOT rewrite the text. Just output the integer IDs.

    Output strictly in JSON format matching this schema:
    {{
        "groups": [[id1, id2], [id3, id4, id5]]
    }}
    If no text needs merging, return: {{"groups": []}}

    OCR Data:
    {json.dumps(llm_payload)}
    """

    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.0,
        max_tokens=2000,
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
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Return standard 8-point polygon: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    return [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]


def reconstruct_text_blocks(minified_data, llm_groups):
    """Step 5 (Failsafe): Reassembles the final JSON, ensuring zero data loss."""
    final_blocks = []
    processed_ids = set()
    
    # Create a lookup dictionary for fast O(1) access
    item_lookup = {item["id"]: item for item in minified_data["items"]}
    
    # Process the LLM's merged groups
    for group in llm_groups.get("groups", []):
        if not group or not isinstance(group, list):
            continue
            
        group_items = []
        for item_id in group:
            if item_id in item_lookup and item_id not in processed_ids:
                group_items.append(item_lookup[item_id])
                processed_ids.add(item_id)
        
        if not group_items:
            continue
            
        # Sort vertically, then horizontally to ensure logical reading order when merged
        group_items.sort(key=lambda item: (item["polygon"][1], item["polygon"][0]))
        
        # Merge text with spaces
        merged_text = " ".join([item["text"] for item in group_items])
        
        # Calculate new encompassing bounding box
        merged_polygon = merge_polygons([item["polygon"] for item in group_items])
        
        # Average the confidence scores
        avg_confidence = sum([item["confidence"] for item in group_items]) / len(group_items)
        
        final_blocks.append({
            "text": merged_text,
            "confidence": round(avg_confidence, 4),
            "polygon": merged_polygon,
            "merged_ids": group
        })
        
    # FAILSAFE: Add all remaining items that the LLM ignored or didn't group
    for item in minified_data["items"]:
        if item["id"] not in processed_ids:
            final_blocks.append({
                "text": item["text"],
                "confidence": item["confidence"],
                "polygon": item["polygon"],
                "merged_ids": [item["id"]]
            })
            
    # Sort final output blocks logically by Y-axis (top to bottom), then X-axis
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
            print(f"   LLM suggested groupings: {llm_groups.get('groups', [])}")
            
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




