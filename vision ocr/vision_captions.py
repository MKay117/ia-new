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

def process_dense_captions():
    for filename in os.listdir("input"):
        file_path = os.path.join("input", filename)
        if not os.path.isfile(file_path): continue

        mime_type, _ = mimetypes.guess_type(file_path)
        print(f"\nExtracting Dense Captions from {filename} (MIME: {mime_type})...")

        with open(file_path, "rb") as f:
            image_data = f.read()

        # Extract only Dense Captions
        result = client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.DENSE_CAPTIONS],
            gender_neutral_caption=True
        )

        output_data = {"metadata": {"width": result.metadata.width, "height": result.metadata.height}, "regions": []}
        
        if result.dense_captions:
            # The first caption is the whole image, subsequent ones are sub-regions
            for caption in result.dense_captions.list:
                output_data["regions"].append({
                    "description": caption.text,
                    "confidence": caption.confidence,
                    "bounding_box": {
                        "x": caption.bounding_box.x, 
                        "y": caption.bounding_box.y, 
                        "w": caption.bounding_box.w, 
                        "h": caption.bounding_box.h
                    }
                })

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_name = f"output/{os.path.splitext(filename)[0]}_dense-captions_{timestamp}.json"
        
        with open(out_name, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    process_dense_captions()