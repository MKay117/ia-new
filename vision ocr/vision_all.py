import os
import time
import json
import mimetypes
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()
ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
KEY = os.getenv("AZURE_VISION_KEY")

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

client = ImageAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

def process_all_features():
    for filename in os.listdir("input"):
        file_path = os.path.join("input", filename)
        if not os.path.isfile(file_path): continue

        mime_type, _ = mimetypes.guess_type(file_path)
        print(f"\nProcessing {filename} (MIME: {mime_type})...")

        with open(file_path, "rb") as f:
            image_data = f.read()

        # Extract all features in one call
        result = client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.TAGS, VisualFeatures.OBJECTS, VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ, 
                VisualFeatures.SMART_CROPS, VisualFeatures.PEOPLE
            ],
            gender_neutral_caption=True
        )

        # Serialize SDK objects to Dictionary
        output_data = {"metadata": {"width": result.metadata.width, "height": result.metadata.height}}
        
        if result.caption:
            output_data["caption"] = {"text": result.caption.text, "confidence": result.caption.confidence}
            
        if result.tags:
            output_data["tags"] = [{"name": t.name, "confidence": t.confidence} for t in result.tags.list]
            
        if result.read:
            output_data["ocr"] = []
            for line in result.read.blocks[0].lines:
                output_data["ocr"].append({
                    "text": line.text,
                    "polygon": [{"x": p.x, "y": p.y} for p in line.bounding_polygon],
                    "words": [{"text": w.text, "confidence": w.confidence} for w in line.words]
                })
                
        if result.dense_captions:
            output_data["dense_captions"] = [
                {"text": c.text, "confidence": c.confidence, "box": {"x": c.bounding_box.x, "y": c.bounding_box.y, "w": c.bounding_box.w, "h": c.bounding_box.h}} 
                for c in result.dense_captions.list
            ]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"output/{os.path.splitext(filename)[0]}_all-vision_{timestamp}.json"
        
        with open(out_name, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    process_all_features()