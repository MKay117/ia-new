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

def process_ocr_only():
    for filename in os.listdir("input"):
        file_path = os.path.join("input", filename)
        if not os.path.isfile(file_path): continue

        mime_type, _ = mimetypes.guess_type(file_path)
        print(f"\nRunning OCR on {filename} (MIME: {mime_type})...")

        with open(file_path, "rb") as f:
            image_data = f.read()

        # Extract only OCR (READ)
        result = client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )

        output_data = {"metadata": {"width": result.metadata.width, "height": result.metadata.height}, "lines": []}
        
        if result.read:
            for line in result.read.blocks[0].lines:
                line_data = {
                    "text": line.text,
                    "bounding_polygon": [{"x": p.x, "y": p.y} for p in line.bounding_polygon],
                    "words": []
                }
                for word in line.words:
                    line_data["words"].append({
                        "text": word.text,
                        "confidence": word.confidence,
                        "bounding_polygon": [{"x": p.x, "y": p.y} for p in word.bounding_polygon]
                    })
                output_data["lines"].append(line_data)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"output/{os.path.splitext(filename)[0]}_ocr_{timestamp}.json"
        
        with open(out_name, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    process_ocr_only()