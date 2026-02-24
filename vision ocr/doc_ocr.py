import os
import json
import time
from datetime import datetime
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

# Configuration [cite: 27, 51]
ENDPOINT = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
KEY = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]
INPUT_DIR = Path("input")
OUTPUT_BASE_DIR = Path("output/docu_intelligence")

def process_architecture_diagrams():
    # Instantiate client [cite: 27]
    client = DocumentIntelligenceClient(ENDPOINT, AzureKeyCredential(KEY))
    
    # Ensure output directory exists
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} not found.")
        return

    for file_path in INPUT_DIR.iterdir():
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf', '.tiff']:
            print(f"Processing: {file_path.name}...")
            
            with open(file_path, "rb") as f:
                # Step 1: Extract Layout (Text, Tables, Selection Marks, Polygons) [cite: 51, 52]
                poller = client.begin_analyze_document("prebuilt-layout", body=f)
                result: AnalyzeResult = poller.result() # Wait for long-running operation 

            # Convert to JSON-serializable dictionary 
            analyze_result_dict = result.as_dict()

            # Define output path: output/docu_intelligence/filename_docs_timestamp.json
            output_filename = f"{file_path.stem}_docs_{timestamp}.json"
            output_path = OUTPUT_BASE_DIR / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f_out:
                json.dump(analyze_result_dict, f_out, indent=4)
            
            print(f"Saved layout data to: {output_path}")

if __name__ == "__main__":
    process_architecture_diagrams()