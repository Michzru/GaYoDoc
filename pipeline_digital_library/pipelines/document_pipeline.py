from ..stages.table_extraction import run_table_extraction
from ..stages.feature_engineering import run_feature_engineering
from ..stages.graph import run_graph_inference
from ..stages.text_detection import run_text_detection
from ..stages.yolo_detection import run_yolo_detection
from ..stages.preprocess_pdf import get_png_images
import json
import os

def run_document_pipeline(pdf_path: str, verbose = True, gpu=True) -> dict:
    print("Starting document pipeline")

    # Stage 0: Convert pdf to png
    pngs, filename = get_png_images(pdf_path, verbose=verbose)

    # Stage 1: YOLO detection
    yolo_detection_results = run_yolo_detection(pngs, filename, verbose=verbose, gpu=gpu)

    # Stage 2: OCR text detection
    text_yolo_detection_results = run_text_detection(pngs, yolo_detection_results, verbose=verbose, gpu=gpu)

    # Stage 3: Prepare graph document/Feature Engineering
    engineered_results = run_feature_engineering(text_yolo_detection_results, verbose=verbose, gpu=gpu)

    # Stage 4: graph (GAT)
    graph_data = run_graph_inference(engineered_results, verbose=verbose, gpu=gpu)

    # Stage 5: table extraction
    final_data = run_table_extraction(graph_data, pngs, verbose=verbose, gpu=gpu)

    # Stage 6: save data [optional]
    # os.makedirs("data/outputs", exist_ok=True)
    #
    # base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # output_filepath = os.path.join("data/outputs", f"{base_name}.json")
    #
    # with open(output_filepath, "w", encoding="utf-8") as f:
    #    json.dump(graph_data, f, indent=4, default=lambda x: None)

    # if verbose:
    #    print(f"Data were successfully saved: {output_filepath}")

    print("Finished document pipeline")

    return final_data