import json
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from paddleocr import PaddleOCR
import os
import joblib
import numpy as np

# Configuration
yolo_filepath = 'models/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt'
pdf_path = "data/Document.pdf"
cache_path = "data/cached_images.joblib"
output_json_path = "data/output_json.json"

# Load Models
print("Loading YOLO model...")
model = YOLOv10(yolo_filepath)

# use_angle_cls=True - finds text if it is upside down
print("Loading OCR model...")
ocr = PaddleOCR(use_angle_cls=True, lang='sk')

# Helper Function to convert pdf into images
def get_pdf_images(pdf_path, cache_path):
    if os.path.exists(cache_path):
        print("Loading images from cache...")
        return joblib.load(cache_path)

    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=300)

    # Compress slightly to save disk space if needed (compress=3)
    joblib.dump(images, cache_path)
    return images

# Helper function to calculate iou over two boxes
def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / float(box1_area + box2_area - intersection_area)

# Get images from the file
images = get_pdf_images(pdf_path, cache_path)

# Data Structure
document_data = {
    "metadata": {
        "filename": os.path.basename(pdf_path),
        "total_pages": len(images)
    },
    "pages": []
}

# Run YOLO on each page
for page_idx, pil_image in enumerate(images):
    print(f"\nProcessing page {page_idx + 1}/{len(images)}...")
    page_width, page_height = pil_image.size

    page_data = {
        "page_number": page_idx + 1,
        "width": page_width,
        "height": page_height,
        "nodes": [],
        "edges": []
    }

    # Perform prediction
    yolo_results = model.predict(
        source=pil_image,
        imgsz=1120,
        conf=0.2,
        #iou=0.1,
        #agnostic_nms=True,
        device="cpu"
    )
    # Grab the results for this specific image
    yolo_result = yolo_results[0]

    valid_boxes = []

    # Prepare boxes
    for box in yolo_result.boxes:
        # Get coordinates in xyxy format
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        category_id = int(box.cls[0].item())
        class_name = yolo_result.names[category_id]
        confidence = box.conf[0].item()

        # Remove QR code in left corner
        is_in_left_zone = x2 < (page_width * 0.1)
        is_in_bottom_zone = y1 > (page_height * 0.90)

        if class_name == "Picture" and is_in_left_zone and is_in_bottom_zone:
            print(f" -> Ignoring image/QR code in bottom left corner: {x1:.0f}, {y1:.0f})")
            continue

        valid_boxes.append({
            "coords": [x1, y1, x2, y2],
            "class_id": category_id,
            "class_name": class_name,
            "confidence": confidence
        })

    filtered_boxes = []
    for i, current_box in enumerate(valid_boxes):
        keep = True
        for j, other_box in enumerate(valid_boxes):
            if i != j:
                iou = calculate_iou(current_box["coords"], other_box["coords"])
                # We keep the one with more confidence if iou is 50%
                if iou > 0.50 and other_box["confidence"] > current_box["confidence"]:
                    keep = False
                    break
        if keep:
            filtered_boxes.append(current_box)

        x1, y1, x2, y2 = current_box["coords"]
        category_id = int(current_box["class_id"])
        class_name = current_box["class_name"]

    # Full page ocr, faster and more accurate
    page_image_np = np.array(pil_image)
    full_page_ocr_results = ocr.ocr(page_image_np, cls=True)

    # Make node and map text to correct node
    for node_id, box_data in enumerate(filtered_boxes):
        x1, y1, x2, y2 = box_data["coords"]

        # Normalization of coords 0 to 1
        norm_x1 = x1 / page_width
        norm_y1 = y1 / page_height
        norm_x2 = x2 / page_width
        norm_y2 = y2 / page_height

        # Center and params of the rectangle
        center_x = (norm_x1 + norm_x2) / 2
        center_y = (norm_y1 + norm_y2) / 2
        width = norm_x2 - norm_x1
        height = norm_y2 - norm_y1

        extracted_text = ""

        #if box_data["class_name"] not in ["Picture"]:
        node_texts = []

        # Mapping text to yolo extraction boxes
        if full_page_ocr_results and full_page_ocr_results[0] is not None:
            for line in full_page_ocr_results[0]:
                ocr_box = line[0]
                text = line[1][0]

                ocr_x_coords = [pt[0] for pt in ocr_box]
                ocr_y_coords = [pt[1] for pt in ocr_box]
                ocr_center_x = sum(ocr_x_coords) / 4.0
                ocr_center_y = sum(ocr_y_coords) / 4.0

                margin = 10
                if (x1 - margin <= ocr_center_x <= x2 + margin) and \
                        (y1 - margin <= ocr_center_y <= y2 + margin):
                    node_texts.append(text)

            extracted_text = " ".join(node_texts)

        node = {
            "node_id": node_id,
            "label": box_data["class_name"],
            "label_id": box_data["class_id"],
            "yolo_confidence": box_data["confidence"],
            "text": extracted_text.strip(),
            "geometry": {
                "absolute_pixel_coords": [int(x1), int(y1), int(x2), int(y2)],
                "normalized_coords": [norm_x1, norm_y1, norm_x2, norm_y2],
                "normalized_center": [center_x, center_y],
                "normalized_size": [width, height]
            }
        }
        page_data["nodes"].append(node)

    document_data["pages"].append(page_data)

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(document_data, f, ensure_ascii=False, indent=4)

print(f"\nDone! JSON generated, saved in: {output_json_path}")