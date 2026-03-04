from shapely.geometry import Polygon, Point, LineString

def run_stage4_stitch_shapely(spatial_data, distance_threshold=75):
    print("-> [Stage 4] Shapely: Stitching Spatial Map into Knowledge Graph...")
    
    nodes = []
    edges = []
    
    # 1. Parse text into Shapely Polygons
    text_polys = {}
    for item in spatial_data.get("spatial_text", []):
        coords = item.get("polygon", [])
        if len(coords) == 8:
            # Convert [x1,y1, x2,y2, x3,y3, x4,y4] to [(x1,y1), (x2,y2)...]
            poly = Polygon([(coords[i], coords[i+1]) for i in range(0, 8, 2)])
            text_polys[item["id"]] = {"text": item["text"], "shape": poly}
            nodes.append({"id": item["id"], "text": item["text"]})

    # 2. Parse Arrowheads into Shapely Points
    arrow_points = [Point(a["point"][0], a["point"][1]) for a in spatial_data.get("detected_arrowheads", [])]

    # 3. Stitch Lines to Nearest Polygons
    for line_idx, line in enumerate(spatial_data.get("detected_connectors", [])):
        x1, y1, x2, y2 = line
        p1, p2 = Point(x1, y1), Point(x2, y2)
        
        # Find nearest text to Point 1
        closest_p1_id, dist_p1 = None, float('inf')
        for t_id, t_data in text_polys.items():
            dist = p1.distance(t_data["shape"])
            if dist < dist_p1:
                closest_p1_id, dist_p1 = t_id, dist
                
        # Find nearest text to Point 2
        closest_p2_id, dist_p2 = None, float('inf')
        for t_id, t_data in text_polys.items():
            dist = p2.distance(t_data["shape"])
            if dist < dist_p2:
                closest_p2_id, dist_p2 = t_id, dist

        # Ignore lines that are too far from any text (likely noise)
        if dist_p1 > distance_threshold or dist_p2 > distance_threshold:
            continue
            
        # Ignore lines that map to the same node (dots/noise inside a box)
        if closest_p1_id == closest_p2_id:
            continue

        # 4. Determine Direction using Arrowheads
        # Check if an arrowhead is closer to p1 or p2
        arrow_at_p1 = any(p1.distance(ap) < 30 for ap in arrow_points)
        arrow_at_p2 = any(p2.distance(ap) < 30 for ap in arrow_points)

        source, target = closest_p1_id, closest_p2_id
        direction = "unidirectional"
        
        if arrow_at_p1 and not arrow_at_p2:
            source, target = closest_p2_id, closest_p1_id # p1 is target
        elif arrow_at_p1 and arrow_at_p2:
            direction = "bidirectional"

        edges.append({
            "connector_id": f"line_{line_idx}",
            "source_id": source,
            "source_text": text_polys[source]["text"],
            "target_id": target,
            "target_text": text_polys[target]["text"],
            "direction": direction
        })

    return {"nodes": nodes, "edges": edges}
