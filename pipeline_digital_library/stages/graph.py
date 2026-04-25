from ..models.gat import get_gat_model
import torch
from torch_geometric.data import Data
from config import CLASS_NAMES
from tqdm import tqdm

def run_graph_inference(data, verbose, gpu):
    model = get_gat_model(verbose=verbose, gpu=gpu)

    if gpu:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model.eval()

    pages = data["pages"]

    iterator = tqdm(
        enumerate(pages),
        total=len(pages),
        desc="Graph inference",
        disable=not verbose,
        leave=False
    )

    with torch.no_grad():
        for page_idx, page in iterator:
            if "feat_geom" not in page:
                continue

            if verbose:
                iterator.set_postfix(page=page_idx + 1)

            feat_geom = torch.as_tensor(page["feat_geom"], dtype=torch.float32)
            feat_yolo = torch.as_tensor(page["feat_yolo"], dtype=torch.float32)
            feat_text = torch.as_tensor(page["feat_text"], dtype=torch.float32)
            edge_index = torch.as_tensor(page["edge_index"], dtype=torch.long)

            batch = Data(
                feat_geom=feat_geom,
                feat_yolo=feat_yolo,
                feat_text=feat_text,
                edge_index=edge_index
            ).to(device)

            # Inference of nodes
            z, node_logits, _ = model(batch)
            pred_classes = node_logits.argmax(dim=-1)
            node_probs = torch.softmax(node_logits, dim=-1)

            node_predicted_classes = {}
            for i, node in enumerate(page["nodes"]):
                class_id = int(pred_classes[i].item())
                class_name = CLASS_NAMES[class_id]
                node["predicted_label"] = class_name
                node["predicted_label_id"] = class_id
                node["predicted_confidence"] = round(float(node_probs[i, class_id].item()), 4)
                node_predicted_classes[node["node_id"]] = class_id

            # Prediction of edges based on gat
            cap_idx = (pred_classes == 0).nonzero(as_tuple=True)[0]
            partner_idx = ((pred_classes == 1) | (pred_classes == 2)).nonzero(as_tuple=True)[0]

            page["edges"] = []

            if len(cap_idx) > 0 and len(partner_idx) > 0:
                # Kartézsky súčin
                src = cap_idx.repeat_interleave(len(partner_idx))
                dst = partner_idx.repeat(len(cap_idx))
                edge_index_targets = torch.stack([src, dst], dim=0)

                # 3. Predikcia hrán (pošleme len kandidátov)
                # POZNÁMKA: Ak tvoja metóda predict_edges vyžaduje aj feat_geom, pridaj ho tam
                edge_logits = model.predict_edges(z, edge_index_targets, batch.feat_geom)

                # Keďže ide o 3-class klasifikáciu (0=nič, 1=cap-fig, 2=cap-tab)
                edge_preds = edge_logits.argmax(dim=-1)
                edge_probs = torch.softmax(edge_logits, dim=-1)

                for i in range(edge_preds.shape[0]):
                    edge_type = int(edge_preds[i].item())

                    if edge_type > 0:  # Ak to GAT spojil (1 alebo 2)
                        src_node_idx = int(edge_index_targets[0, i].item())
                        dst_node_idx = int(edge_index_targets[1, i].item())

                        # Získame skutočné ID uzlov z JSONu
                        source_id = page["nodes"][src_node_idx]["node_id"]
                        target_id = page["nodes"][dst_node_idx]["node_id"]

                        page["edges"].append({
                            "source": source_id,
                            "target": target_id,
                            "relation_type": edge_type,  # 1=cap-fig, 2=cap-tab
                            "confidence": round(float(edge_probs[i, edge_type].item()), 4)
                        })


            page.pop("feat_geom", None)
            page.pop("feat_yolo", None)
            page.pop("feat_text", None)
            page.pop("edge_index", None)
    return data