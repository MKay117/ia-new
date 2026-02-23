import os
import time
import json
import base64
import mimetypes
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load credentials
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_VISION_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure directories exist
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

def get_mime_type(file_path):
    """Step 1: Determine MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def run_azure_ocr(image_path):
    """Step 2: Fetch text, polygons, and metadata via Azure."""
    url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_KEY,
        'Content-Type': 'application/octet-stream'
    }
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = requests.post(url, headers=headers, data=image_data)
    response.raise_for_status()
    operation_url = response.headers["Operation-Location"]
    
    # Poll for results
    while True:
        poll_response = requests.get(operation_url, headers={'Ocp-Apim-Subscription-Key': AZURE_KEY})
        result = poll_response.json()
        if result['status'] not in ['notStarted', 'running']:
            break
        time.sleep(1)
        
    return result

def minify_ocr_data(raw_azure_data):
    """Minifies raw Azure JSON into a lightweight payload."""
    minified_payload = {
        "metadata": {},
        "lines": []
    }
    
    if raw_azure_data['status'] == 'succeeded':
        read_result = raw_azure_data['analyzeResult']['readResults'][0]
        
        # Image Metadata
        minified_payload["metadata"] = {
            "width": read_result.get('width'),
            "height": read_result.get('height'),
            "angle": read_result.get('angle')
        }
        
        # Text, Polygons, and Confidence
        for line in read_result.get('lines', []):
            words = line.get('words', [])
            confidences = [w.get('confidence', 1.0) for w in words if 'confidence' in w]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
            
            minified_payload["lines"].append({
                "text": line['text'],
                "polygon": line['boundingBox'], 
                "confidence": round(avg_confidence, 4)
            })
            
    return minified_payload

def encode_image_base64(image_path):
    """Encodes image for the LLM."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_llm_correction(image_path, mime_type, minified_data):
    """Step 4: Send Image and Minified Data to GPT-4o."""
    base64_image = encode_image_base64(image_path)
    
    # Explicitly asking for JSON is required by OpenAI's API when using json_object mode
    prompt = f"""
    You are an expert OCR correction assistant. 
    I am providing you with an architecture diagram and the raw OCR data extracted from it.

    The raw OCR data often incorrectly separates multi-line labels (e.g., splitting "Microsoft" and "Windows" into two items even though they form one visual label). 
    Your task is to:
    1. Look at the image to understand the visual layout of the text.
    2. Use the provided OCR bounding box data to accurately group text that visually belongs together as a single label, title, or block of text. 
    3. If text is floating or inside a boundary box, group it logically based on its visual proximity and alignment.
    4. Return a JSON object with a single key 'text_blocks', containing an array of objects. Each object must have 'text' (the corrected, combined string), 'confidence' (averaged if combined), and 'polygon' (a new bounding box that encompasses all combined text).

    Raw OCR Data:
    {json.dumps(minified_data)}
    """

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    
    return json.loads(response.choices[0].message.content)

def main():
    for filename in os.listdir("input"):
        file_path = os.path.join("input", filename)
        
        if not os.path.isfile(file_path):
            continue
            
        print(f"\n--- Processing: {filename} ---")
        base_name = os.path.splitext(filename)[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Step 1: MIME Type
        mime_type = get_mime_type(file_path)
        print(f"Detected MIME type: {mime_type}")
        
        # Step 2: Azure OCR
        print("Running Azure OCR (Stage 1)...")
        stage_1_output = run_azure_ocr(file_path)
        
        # Step 3: Save Stage 1 Output
        stage_1_filename = f"output/{base_name}_stage_1_{timestamp}.json"
        with open(stage_1_filename, 'w') as f:
            json.dump(stage_1_output, f, indent=4)
        print(f"Saved raw OCR to: {stage_1_filename}")
        
        # Minify for LLM
        minified_data = minify_ocr_data(stage_1_output)
        
        print("\n--- Minified JSON Payload for LLM ---")
        print(json.dumps(minified_data, indent=2))
        print("-------------------------------------\n")
        
        # Step 4: LLM Correction
        print("Running GPT-4o Correction (Stage 2)...")
        try:
            stage_2_output = run_llm_correction(file_path, mime_type, minified_data)
            
            # Step 5: Save Stage 2 Output
            stage_2_filename = f"output/{base_name}_stage_2_{timestamp}.json"
            with open(stage_2_filename, 'w') as f:
                json.dump(stage_2_output, f, indent=4)
            print(f"Saved LLM Output to: {stage_2_filename}")
            
        except Exception as e:
            print(f"LLM Processing failed for {filename}: {e}")

if __name__ == "__main__":
    main()
