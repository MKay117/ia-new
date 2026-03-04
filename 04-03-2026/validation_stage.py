def run_stage5_vlm_validation(image_path, draft_graph):
    print("-> [Stage 5] Azure OpenAI: Strict VLM Graph Validation...")
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    base64_image = encode_image(image_path)
    draft_json_str = json.dumps(draft_graph, indent=2)

    prompt = f"""
    Role: You are a strict QA Graph Auditor for infrastructure architectures.

    Inputs Provided:
    1. Image: The original architecture diagram.
    2. Draft JSON: A preliminary knowledge graph generated via coordinate math.
    
    Draft JSON:
    {draft_json_str}

    Your Task:
    Audit the Draft JSON against the Image with absolute strictness. Output a final, corrected JSON.

    Validation Rules:
    1. Prune False Positives (No Hallucinations): If an edge or node exists in the Draft JSON but does not visually exist in the Image, DELETE it. Do not assume connections.
    2. Add False Negatives (Missing Data): If a clear visual connection or labeled component exists in the Image but is missing from the Draft JSON, ADD it. Assign a unique ID (e.g., "vlm_added_1") to new nodes.
    3. Correct Directions: Verify the "direction" of every edge. Rely strictly on the visual arrowheads. 
    4. Correct Node Names: Fix any OCR typos in the "text" fields based on the visual labels.

    Output ONLY valid JSON matching this exact structure:
    {{
      "nodes": [ {{ "id": "string", "text": "string" }} ],
      "edges": [ {{ "source_id": "string", "source_text": "string", "target_id": "string", "target_text": "string", "direction": "unidirectional | bidirectional" }} ]
    }}
    """

    try:
        response = client.chat.completions.create(
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0, # Strict adherence, no creative temperature
        )
        
        return json.loads(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error during VLM validation: {e}")
        return draft_graph # Fallback to draft if API fails
