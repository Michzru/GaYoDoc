from ..models.docling import get_docling_converter
from tqdm import tqdm
import io
from docling.datamodel.base_models import DocumentStream

# Docling label_id mapping — fixný, nezávisí od dokumentu
DOCLING_LABEL_TO_ID = {
    "caption":              0,
    "footnote":             1,
    "formula":              2,
    "list_item":            3,
    "page_footer":          4,
    "page_header":          5,
    "picture":              6,
    "section_header":       7,
    "table":                8,
    "text":                 9,
    "title":                10,
    "document_index":       11,
    "code":                 12,
    "checkbox_selected":    13,
    "checkbox_unselected":  14,
    "form":                 15,
    "key_value_region":     16,
}

def run_docling_detection(pngs, filename, verbose, gpu):
    converter = get_docling_converter(verbose=verbose, gpu=gpu)

    document_data = {
        "metadata": {
            "filename": filename,
            "total_pages": len(pngs)
        },
        "pages": []
    }

    iterator = tqdm(
        enumerate(pngs),
        total=len(pngs),
        desc="Docling pages",
        leave=False,
        disable=not verbose
    )

    for page_idx, pil_image in iterator:
        if verbose:
            iterator.set_postfix(page=page_idx + 1)

        page_width, page_height = pil_image.size

        page_data = {
            "page_number": page_idx + 1,
            "width": page_width,
            "height": page_height,
            "nodes": [],
            "edges": []
        }

        # PIL → ByteStream → Docling
        image_io = io.BytesIO()
        pil_image.save(image_io, format="PNG")
        image_io.seek(0)
        doc_stream = DocumentStream(name=f"page_{page_idx + 1}.png", stream=image_io)
        result = converter.convert(doc_stream)

        page = result.pages[0]
        pw = page.size.width  if page.size else page_width
        ph = page.size.height if page.size else page_height

        layout = getattr(getattr(page, "predictions", None), "layout", None)
        if not layout:
            document_data["pages"].append(page_data)
            continue

        # cluster id → text
        text_by_cluster = {}
        for cl in layout.clusters:
            cid = getattr(cl, "id", None)
            if cid is None:
                continue
            if hasattr(cl, "text") and cl.text:
                text_by_cluster[cid] = cl.text.strip()
            elif hasattr(cl, "cells"):
                words = [cell.text for cell in cl.cells if getattr(cell, "text", "")]
                text_by_cluster[cid] = " ".join(words).strip()

        for node_id, cl in enumerate(layout.clusters):
            bbox  = cl.bbox
            label = cl.label.value if hasattr(cl.label, "value") else str(cl.label)
            conf  = float(getattr(cl, "confidence", 0.0))
            cid   = getattr(cl, "id", None)

            x1, y1, x2, y2 = bbox.l, bbox.t, bbox.r, bbox.b

            norm_x1 = x1 / pw
            norm_y1 = y1 / ph
            norm_x2 = x2 / pw
            norm_y2 = y2 / ph

            node = {
                "node_id":   node_id,
                "label":     label,
                "label_id":  DOCLING_LABEL_TO_ID.get(label, 9),  # default → text
                "confidence": round(conf, 6),
                "text":      text_by_cluster.get(cid, ""),
                "geometry": {
                    "absolute_pixel_coords": [int(x1), int(y1), int(x2), int(y2)],
                    "normalized_coords":     [norm_x1, norm_y1, norm_x2, norm_y2],
                    "normalized_center":     [(norm_x1 + norm_x2) / 2, (norm_y1 + norm_y2) / 2],
                    "normalized_size":       [norm_x2 - norm_x1, norm_y2 - norm_y1]
                }
            }
            page_data["nodes"].append(node)

        document_data["pages"].append(page_data)

    return document_data