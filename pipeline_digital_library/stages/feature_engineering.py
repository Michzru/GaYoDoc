from ..models.transformer import get_transformer_model
from ..utils.features import prepare_page_tensors, build_knn_edges
from tqdm import tqdm

def run_feature_engineering(data, verbose, gpu):
    """
    Extracts features and builds the graph structure for each page.
    Does NOT run model inference or prediction.

    Args:
        data (dict): Document data containing pages and nodes.
        verbose (bool): If False, output will be printed to stdout.

    Returns:
        dict: Document data enriched with node features and edges (no predictions).
    """
    model = get_transformer_model(verbose=verbose)
    pages = data["pages"]

    iterator = tqdm(
        enumerate(pages),
        total=len(pages),
        desc="Feature engineering",
        disable=not verbose,
        leave=False
    )

    for page_idx, page in iterator:
        if verbose:
            iterator.set_postfix(page=page_idx + 1)

        nodes = page["nodes"]

        if len(nodes) < 2:
            continue

        image_W = page["width"]
        image_H = page["height"]

        result = prepare_page_tensors(nodes, model, image_W, image_H)

        if result is None:
            continue

        feat_geom, feat_docling, feat_text = result  # zmenené

        edge_index = build_knn_edges(feat_geom, k=8)

        page["feat_geom"]    = feat_geom
        page["feat_docling"] = feat_docling           # zmenené
        page["feat_text"]    = feat_text
        page["edge_index"]   = edge_index

    return data