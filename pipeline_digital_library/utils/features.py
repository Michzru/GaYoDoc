import torch
import re
import numpy as np
import unicodedata
from torch_geometric.data import Data

def remove_diacritics(text: str) -> str:
    """Normalizes text by removing diacritical marks"""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def extract_manual_text_features(text: str, xc: float = 0.5, yc: float = 0.5) -> list:
    """
    Creates 13 hand-crafted text features.
    Features:
      - is_fig:          Is the text a figure caption? (EN/SK/CZ)
      - is_tab:          Is the text a table caption? (EN/SK/CZ)
      - is_num:          Does the text start with a number/equation?
      - is_short:        Is the text short (likely a title/label)?
      - len_norm:        Normalized word count (capped at 1.0)
      - is_top:          Is the element in the top header zone? (yc < 0.15)
      - is_bottom:       Is the element in the bottom footer zone? (yc > 0.85)
      - ends_colon:      Does the text end with a colon?
      - has_section_num: Does the text start with a section number? (e.g. "3.2 Methods")
      - is_allcaps:      Is the text all uppercase?
      - avg_word_len:    Average word length normalized
      - has_code_chars:  Contains code-like characters (=, {, }, ;)
      - has_math:        Contains math operators (+, -, *, /, ^, \)
    """
    t  = text.strip()
    tl = remove_diacritics(t.lower())
    words      = t.split()
    word_count = len(words)

    # --- pôvodných 5 ---
    is_fig = 1.0 if re.match(
        r'^(figure|fig\.?|obrazok|obr\.?)\s*[\d\(\[]', tl
    ) else 0.0

    is_tab = 1.0 if re.match(
        r'^(table|tab\.?|tabulka)\s*[\d\(\[]', tl
    ) else 0.0

    is_num = 1.0 if re.match(
        r'^\(?\d+[\.\)\:]', tl
    ) else 0.0

    is_short = 1.0 if word_count < 20 else 0.0
    len_norm = min(word_count / 100.0, 1.0)

    # --- poziové ---
    is_top    = 1.0 if yc < 0.15 else 0.0
    is_bottom = 1.0 if yc > 0.85 else 0.0

    # --- štruktúra textu ---
    ends_colon      = 1.0 if t.endswith(':') else 0.0
    has_section_num = 1.0 if re.match(r'^\d+(\.\d+)*\s+[A-Z]', t) else 0.0
    is_allcaps      = 1.0 if (t.isupper() and len(t) > 3) else 0.0
    avg_word_len    = min(np.mean([len(w) for w in words]) / 10.0, 1.0) if words else 0.0

    # --- obsah ---
    has_code_chars = 1.0 if re.search(r'[=\{\};]', t) else 0.0
    has_math       = 1.0 if re.search(r'[\+\-\*/\^\\]', t) else 0.0

    return [
        is_fig, is_tab, is_num, is_short, len_norm,   # 5
        is_top, is_bottom,                             # 2
        ends_colon, has_section_num, is_allcaps,       # 3
        avg_word_len, has_code_chars, has_math         # 3
    ]  # celkom 13


def build_knn_edges(feat_geom, k=8):
    """
    Creates graph edges connected with k closest neighbors.
    Utilizes weighted Euclidean distance algorithm to compute the distance
    (prioritizes vertical relationships)
    """
    num_nodes = feat_geom.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    actual_k = min(k, num_nodes - 1)
    centers  = feat_geom[:, 4:6].cpu().numpy()  # .cpu() pre MPS kompatibilitu

    edge_list = []
    for i in range(num_nodes):
        dx        = centers[:, 0] - centers[i, 0]
        dy        = centers[:, 1] - centers[i, 1]
        distances = np.sqrt(dx ** 2 + (1.5 * dy) ** 2)
        nearest   = np.argsort(distances)[1:actual_k + 1]

        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


# Fixný Docling label → label_id mapping
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
N_DOCLING = 17  # počet Docling tried


def prepare_page_tensors(page_nodes, text_embedder, image_W, image_H):
    """
    Takes nodes from one page and returns triplet tensors.
    Each node on the page:
      feat_geom    — [N, 11]  geometry + confidence
      feat_docling — [N, 17]  soft-label distribution of Docling classes
      feat_text    — [N, 397] 13 hand-crafted + 384 transformer features
    """
    if not page_nodes:
        return None

    texts = [node.get("text", "") for node in page_nodes]
    embeddings = text_embedder.encode(
        texts,
        convert_to_tensor=False,
        show_progress_bar=False
    )

    feat_geom, feat_docling, feat_text = [], [], []

    for idx, node in enumerate(page_nodes):
        # --- Geometria (11 hodnôt) ---
        coords = node["geometry"]["absolute_pixel_coords"]
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1

        x1_norm  = x1 / image_W
        y1_norm  = y1 / image_H
        x2_norm  = x2 / image_W
        y2_norm  = y2 / image_H
        x_center = (x1_norm + x2_norm) / 2
        y_center = (y1_norm + y2_norm) / 2
        w_norm   = x2_norm - x1_norm
        h_norm   = y2_norm - y1_norm
        area     = np.log(((w * h) / (image_W * image_H)) + 1e-6)
        aspect   = np.log((w / (h + 1e-6)) + 1e-6)
        conf     = float(node.get("confidence", 1.0))   # docling kľúč

        feat_geom.append([
            x1_norm, y1_norm, x2_norm, y2_norm,
            x_center, y_center, w_norm, h_norm,
            area, aspect, conf
        ])

        # --- Docling soft-label (17-dim) ---
        label_id = node.get("label_id", None)

        docling_soft = np.full(N_DOCLING, (1.0 - conf) / (N_DOCLING - 1), dtype=np.float32)
        if label_id is not None and 0 <= int(label_id) < N_DOCLING:
            docling_soft[int(label_id)] = conf
        else:
            docling_soft = np.full(N_DOCLING, 1.0 / N_DOCLING, dtype=np.float32)

        docling_soft *= 0.3
        feat_docling.append(docling_soft.tolist())

        # --- Text features (13 + 384 = 397) ---
        manual_feats     = extract_manual_text_features(texts[idx], xc=x_center, yc=y_center)
        transformer_feats = embeddings[idx].tolist()
        feat_text.append(manual_feats + transformer_feats)

    return (
        torch.tensor(feat_geom,    dtype=torch.float),
        torch.tensor(feat_docling, dtype=torch.float),
        torch.tensor(feat_text,    dtype=torch.float),
    )