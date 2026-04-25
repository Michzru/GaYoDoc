# YoGaDoc v0.0.1

A document processing pipeline that extracts structured information from PDF files using layout detection, OCR, graph neural network, and table structure recognition.

## Installation

```bash
pip install -e .
```

## Download Model Weights

Place all weights in the `model_weights/` directory at the project root.

**DocLayout-YOLO** — run the provided download script:
```bash
python scripts/download_yolo_model.py
```

**GAT model** (`best_model.pt`) — included in the repository under `model_weights/`.

**UniTable** — download `unitable_large_structure.pt` and `unitable_large_bbox.pt` from:
- https://huggingface.co/poloclub/UniTable/tree/main
- or https://github.com/poloclub/unitable

Place them in `model_weights/unitable/`.

After setup, your `model_weights/` directory should look like this:

```
model_weights/
├── doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt
├── best_model.pt
└── unitable/
    ├── unitable_large_structure.pt
    └── unitable_large_bbox.pt
```

## Usage

```python
from pipeline_digital_library import run_pipeline

result = run_pipeline("path/to/document.pdf", pipeline="document", gpu=True, verbose=True)
```

### Parameters

- `file_path` — path to the input PDF
- `pipeline` — pipeline type, currently `"document"`
- `gpu` — use GPU acceleration if available (MPS on Apple Silicon, CUDA otherwise)
- `verbose` — print progress to stdout

## Pipeline Stages

The pipeline processes a PDF through six sequential stages:

**Stage 0 — PDF to PNG**  
Converts each page of the PDF to a high-resolution PNG image.

**Stage 1 — Layout Detection (YOLO)**  
Runs DocLayout-YOLO to detect document elements on each page and assigns them an initial label from the DocLayNet taxonomy: `Caption`, `Picture`, `Table`, `Formula`, `Section-header`, `Page-footer`, `Page-header`, and more.

**Stage 2 — OCR (EasyOCR)**  
Performs full-page OCR and assigns extracted text to each detected node.

**Stage 3 — Feature Engineering**  
Computes geometric and semantic features for each node in preparation for graph inference. Includes normalized coordinates, spatial relationships, and text embeddings via `sentence-transformers/all-MiniLM-L6-v2`.

**Stage 4 — Graph Inference (GAT)**  
A Graph Attention Network reclassifies each node and predicts directed edges between nodes. This captures relationships such as caption → figure and caption → table. Each edge has an associated confidence score.

**Stage 5 — Table Extraction (UniTable)**  
For nodes classified as `Table`, runs UniTable to recover the full HTML structure and bounding boxes of cells.

## Output Format

The pipeline returns a dictionary with the following structure:

```json
{
  "metadata": {
    "filename": "document.pdf",
    "total_pages": 5
  },
  "pages": [
    {
      "page_number": 1,
      "width": 2103,
      "height": 3000,
      "nodes": [
        {
          "node_id": 0,
          "label": "Text",
          "yolo_confidence": 0.98,
          "text": "Extracted text content...",
          "geometry": {
            "absolute_pixel_coords": [x1, y1, x2, y2],
            "normalized_coords": [x1, y1, x2, y2],
            "normalized_center": [cx, cy],
            "normalized_size": [w, h]
          },
          "predicted_label": "Section-header",
          "predicted_label_id": 4,
          "predicted_confidence": 0.96,
          "table_data": null
        }
      ],
      "edges": [
        {
          "source": 0,
          "target": 1,
          "confidence": 0.94
        }
      ]
    }
  ]
}
```

### Node Fields

| Field | Description |
|-------|-------------|
| `node_id` | Index of the node on its page |
| `label` | Original YOLO label |
| `yolo_confidence` | YOLO detection confidence |
| `text` | OCR-extracted text |
| `geometry` | Bounding box in absolute pixels and normalized coordinates |
| `predicted_label` | GAT reclassified label |
| `predicted_label_id` | Numeric ID of the GAT label |
| `predicted_confidence` | GAT classification confidence |
| `table_data` | HTML table structure from UniTable (only for Table nodes) |

### Edge Fields

Edges are directed and represent relationships predicted by the GAT model (e.g. a caption pointing to its associated figure or table).

| Field | Description |
|-------|-------------|
| `source` | `node_id` of the source node |
| `target` | `node_id` of the target node |
| `confidence` | Edge prediction confidence |

## Project Structure

```
pipeline_digital_library/
├── pipeline_digital_library/   # Main package
│   ├── models/                 # Model loading (YOLO, GAT, UniTable, Transformer)
│   ├── pipelines/              # Pipeline orchestration
│   ├── stages/                 # Individual processing stages
│   ├── utils/                  # Shared utilities
│   ├── config.py               # Paths and constants
│   └── core.py                 # Public API entry point
├── model_weights/              # Downloaded model checkpoints (not included)
├── data/                       # Input documents
├── scripts/                    # Download and Visualization Scripts
└── pyproject.toml
```

## Requirements

See `requirements.txt` or `pyproject.toml` for the full dependency list. Key dependencies include `torch`, `doclayout-yolo`, `easyocr`, `torch-geometric`, `sentence-transformers`, and `pdf2image`.

Poppler is required for PDF rendering:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt install poppler-utils
```

## Credits

This pipeline builds on the following open-source projects:

- **[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)** — document layout detection
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — optical character recognition  
- **[UniTable](https://github.com/poloclub/unitable)** — table structure recognition
- **[sentence-transformers](https://github.com/UKPLab/sentence-transformers)** — text embeddings (`all-MiniLM-L6-v2`)
