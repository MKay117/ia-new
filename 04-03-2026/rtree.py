import math
from rtree import index

def get_bbox(polygon_list):
    # Extracts minX, minY, maxX, maxY from an 8-point flat list
    xs = polygon_list[0::2]
    ys = polygon_list[1::2]
    return (min(xs), min(ys), max(xs), max(ys))

def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def run_stage4_stitch_rtree(spatial_data, distance_threshold=75):
    print("-> [Stage 4] Rtree: Stitching Spatial Map into Knowledge Graph...")
    
    nodes = []
    edges = []
    
    # 1. Build Spatial Index
    idx = index.Index()
    text_lookup = {}
    
    for i, item in enumerate(spatial_data.get("spatial_text", [])):
        coords = item.get("polygon", [])
        if len(coords) == 8:
            bbox = get_bbox(coords)
            idx.insert(i, bbox) # Insert into Rtree
            text_lookup[i] = {"id": item["id"], "text": item["text"], "bbox": bbox}
            nodes.append({"id": item["id"], "text": item["text"]})

    arrowheads = spatial_data.get("detected_arrowheads", [])

    # 2. Stitch Lines
    for line_idx, line in enumerate(spatial_data.get("detected_connectors", [])):
        x1, y1, x2, y2 = line
        
        # Query Rtree for nearest bounding box to endpoints
        # nearest() returns a generator of the closest IDs
        nearest_to_p1 = list(idx.nearest((x1, y1, x1, y1), 1))
        nearest_to_p2 = list(idx.nearest((x2, y2, x2, y2), 1))
        
        if not nearest_to_p1 or not nearest_to_p2:
            continue
            
        n1_data = text_lookup[nearest_to_p1[0]]
        n2_data = text_lookup[nearest_to_p2[0]]

        if n1_data["id"] == n2_data["id"]:
            continue # Self-loop / noise

        # 3. Determine Direction using basic math distance
        arrow_at_p1 = False
        arrow_at_p2 = False
        
        for ah in arrowheads:
            ax, ay = ah["point"]
            if euclidean_dist(x1, y1, ax, ay) < 30: arrow_at_p1 = True
            if euclidean_dist(x2, y2, ax, ay) < 30: arrow_at_p2 = True

        source_data, target_data = n1_data, n2_data
        direction = "unidirectional"

        if arrow_at_p1 and not arrow_at_p2:
            source_data, target_data = n2_data, n1_data
        elif arrow_at_p1 and arrow_at_p2:
            direction = "bidirectional"

        edges.append({
            "connector_id": f"line_{line_idx}",
            "source_id": source_data["id"],
            "source_text": source_data["text"],
            "target_id": target_data["id"],
            "target_text": target_data["text"],
            "direction": direction
        })

    return {"nodes": nodes, "edges": edges}
