import os
import time
from datetime import datetime
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

from openai import AzureOpenAI

load_dotenv()

# Config
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


INPUT_DIR = Path("input")
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_arrow_connections(image_path):
    print("-> [Stage 1] GPT-4o Vision: Analyzing Legend & Extracting it...")
    llm_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    prompt = """
        Your are an experienced Principal Cloud Architect in a BFSI organisation.
        Analyze this infrastructure architecture diagram and check whether it has any legend or not.
        If there is a legend, understand and give it in below mentioned format.
        If there is no legend infer it from standard cloud architecure practices and give it in below mentioned format.
        Clearly identify the color and type of line - solid or dash or single direction arrow or bi directional arrow or double lines. Basically all possible line formats across all applications.
        JSON output format: {
            "legend": {
                "line_type": <dashed> | <solid> | <single direction arrow> | <bi directional arrow>,
                "line_color": <color of lines>,
            }
        }
    """

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content)


def extract_icons(image_path):
    print("-> [Stage 3] Consolidating into Final Architectural JSON...")
    llm_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    prompt = """
        You are an experienced Principal Cloud Architect in a BFSI organisation.
        Refer the give image, understand all icons from standard market practices and give the list of all icons is a proper json format with names
        
        JSON output format: {
            "icons": {
                "icon_name": <name of the icon>,
                "finding_method": <name near or around icon> | inference | assumption | <common knowledge about icon>,
                "assumption": <If finding method is assumption, where did the assumption come from ? thought process explanation>,
                "inference": <If finding method is inference, where did the inference come from ? thought process explanation>,
                "confidence_score": <your confidence percentage regarding the icon>, # higher score means accurate (80% to 100%), least score means leat accurate (0 - 20%), medium score (20% to 80%) means moderate and needs explanation 
                "confidence_explanation": <explanation for confidence score and any issues with image or icon in giving that score>
            }
        }
    """

    response = llm_client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                ],
            }
        ],
    )
    return json.loads(response.choices[0].message.content)


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f"output/lmm test/{date_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in INPUT_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            continue

        print(f"\n========== Processing: {img_path.name} ==========")
        timestamp = time.strftime("%H%M%S")
        base_name = img_path.stem

        # Stage 1: Fetching Arrows
        legend = extract_arrow_connections(img_path)

        step1_path = (
            output_dir / f"{base_name}_step1_arrow_connections_{timestamp}.json"
        )
        with open(step1_path, "w") as f:
            json.dump(legend, f, indent=4)

        # Stage 2: Fetching Icons
        icons = extract_icons(img_path)

        step2_icons = output_dir / f"{base_name}_step2_icons_{timestamp}.json"
        with open(step2_icons, "w") as f:
            json.dump(icons, f, indent=4)

        print(f"-> SUCCESS: Files saved to {output_dir}")


if __name__ == "__main__":
    main()


# "text_on_line": <any text mentioned on or above or below the line>,
#                 "ports_on_line": <any ports mentioned on or above or below the line>,
#                 "cidr_on_line" <any cidr mentioned on or above or below the line>,
#                 "source": <name or text at the source / starting point>,
#                 "stages": [stages or components the line is passing through], # stages should be in list. ex: ["source_name", "stage_1", "stage_2", "stage_3", "target"]
#                 "target": <name or text at the target / ending point>,
