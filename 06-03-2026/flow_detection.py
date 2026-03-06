import cv2
import numpy as np
import math
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

def create_exclusion_mask(img_shape, spatial_data):
    """Creates a mask to blot out text and complex icons so lines don't tangle in them."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Fill text polygons
    for item in spatial_data.get("spatial_text", []):
        pts = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
        if len(pts) > 0:
            cv2.fillPoly(mask, [pts], 255)
            
    # Fill icon polygons
    for item in spatial_data.get("icons", []):
        pts = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
        if len(pts) > 0:
            cv2.fillPoly(mask, [pts], 255)
            
    return mask

def run_stage2_advanced_geometry(image_path, spatial_data):
    print("-> [Stage 2] Advanced Geometry: Masking, Bridging, and Skan Tracing...")
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Binarize: Lines become White (255), background Black (0)
    # Using adaptive thresholding for clean digital images
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Apply Exclusion Mask
    # Subtract the areas where text/icons exist so we ONLY trace lines
    exclusion_mask = create_exclusion_mask(img.shape, spatial_data)
    lines_only = cv2.bitwise_and(binary, cv2.bitwise_not(exclusion_mask))
    
    # 3. Morphological Closing (Bridge dashed lines and thick tunnels)
    # A 5x5 kernel smudges pixels up to 5 pixels away, connecting dashed gaps
    kernel = np.ones((5,5), np.uint8)
    bridged_lines = cv2.morphologyEx(lines_only, cv2.MORPH_CLOSE, kernel)
    
    # 4. Skeletonize (Reduce to 1-pixel paths)
    # skeletonize requires a boolean array
    bool_skeleton = skeletonize(bridged_lines > 0)
    
    # 5. Extract Paths with Skan
    branch_data = summarize(Skeleton(bool_skeleton))
    
    extracted_lines = []
    # skan provides coordinates for the entire path, not just start/end
    for _, row in branch_data.iterrows():
        # Ignore micro-noise artifacts (< 15 pixels long)
        if row['branch-distance'] < 15:
            continue
            
        # Get start and end points
        y1, x1 = int(row['image-coord-src-0']), int(row['image-coord-src-1'])
        y2, x2 = int(row['image-coord-dst-0']), int(row['image-coord-dst-1'])
        
        extracted_lines.append({
            "start": [x1, y1],
            "end": [x2, y2],
            "euclidean_length": row['euclidean-distance'],
            "path_length": row['branch-distance'] # If path > euclidean, it bends!
        })

    # 6. Arrowhead Detection (Shi-Tomasi Corners)
    # We look for sharp corners on the original gray image
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.05, minDistance=5)
    arrowheads = []
    
    if corners is not None:
        corners = np.intp(corners)
        for corner in corners:
            cx, cy = corner.ravel()
            
            # Filter: Is this corner near the endpoint of our extracted lines?
            for line in extracted_lines:
                dist_start = math.hypot(cx - line["start"][0], cy - line["start"][1])
                dist_end = math.hypot(cx - line["end"][0], cy - line["end"][1])
                
                # If a sharp corner is within a 20px radius of a line ending, it's an arrow
                if dist_start < 20 or dist_end < 20:
                    arrowheads.append({"point": [int(cx), int(cy)]})
                    break # Don't count the same arrowhead twice

    return {
        "lines": extracted_lines,
        "arrowheads": arrowheads
    }
