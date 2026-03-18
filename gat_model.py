import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class GATMultitask(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)

        self.node_head = nn.Linear(hidden_channels, 2)

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    # def forward(self, x, edge_index):
    #     x = F.elu(self.conv1(x, edge_index))
    #     x = F.dropout(x, p=0.2, training=self.training)
    #     z = self.conv2(x, edge_index)

    #     node_logits = self.node_head(z)
    #     return z, node_logits
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        z = self.conv2(x, edge_index)
        return z

    # def predict_edges(self, z, edge_index):
    #     # z[edge_index[0]] sú vektory "odkiaľ", z[edge_index[1]] sú vektory "kam"
    #     #return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    #     edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
    #     return self.edge_head(edge_features).squeeze(-1)

    def predict_edges(self, z, edge_index, x_geom):
        pos_src = x_geom[edge_index[0], :2]
        pos_dst = x_geom[edge_index[1], :2]

        dist = torch.norm(pos_src - pos_dst, dim=-1, keepdim=True)

        edge_features = torch.cat([z[edge_index[0]], z[edge_index[1]], dist], dim=-1)

        return self.edge_head(edge_features).squeeze(-1)


def extract_features_for_gat(page_nodes):
    features = []

    for node in page_nodes:
        x_center, y_center = node["geometry"]["normalized_center"]
        w_norm, h_norm = node["geometry"]["normalized_size"]

        area = w_norm * h_norm
        aspect = w_norm / (h_norm + 1e-6)

        category_id = np.zeros(11)
        idx = min(node["label_id"], 10)
        category_id[idx] = 1

        geom_features = np.array([x_center, y_center, w_norm, h_norm, area, aspect])
        feature = np.concatenate([geom_features, category_id])
        features.append(feature)

    if not features:
        return torch.empty((0, 17), dtype=torch.float)

    return torch.tensor(np.array(features), dtype=torch.float)


def build_knn_edges_inference(features, k=8):
    num_nodes = features.shape[0]
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    actual_k = min(k, num_nodes - 1)
    centers = features[:, :2].numpy()  # x_center, y_center
    edge_list = []

    for i in range(num_nodes):
        dx = centers[:, 0] - centers[i, 0]
        dy = centers[:, 1] - centers[i, 1]

        distances = np.sqrt(dx ** 2 + (1.5 * dy) ** 2)
        nearest = np.argsort(distances)[1:actual_k + 1]

        for j in nearest:
            edge_list.append([i, j])
            edge_list.append([j, i])

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = GATMultitask(in_channels=17, hidden_channels=64)
model.load_state_dict(torch.load('models/gat_model_version_01_0.92_f1.pt', map_location=device))
model.to(device)
model.eval()


json_path = "data/output_json.json"
with open(json_path, 'r', encoding='utf-8') as f:
    document_data = json.load(f)


with torch.no_grad():
    for page in document_data["pages"]:
        if len(page["nodes"]) < 2:
            continue

        print(f"Processing page {page['page_number']}...")

        x = extract_features_for_gat(page["nodes"]).to(device)

        edge_index = build_knn_edges_inference(x.cpu(), k=8).to(device)

        z = model(x, edge_index)

        edge_logits = model.predict_edges(z, edge_index, x)

        edge_probs = torch.sigmoid(edge_logits)

        predicted_edges = []
        for i in range(len(edge_probs)):
            prob = edge_probs[i].item()

            if prob > 0.3:
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())

                predicted_edges.append({
                    "source": src,
                    "target": dst,
                    "probability": round(prob, 4)
                })

        page["edges"] = predicted_edges
        print(f" -> Found {len(predicted_edges)} true edges.")


output_json = "data/document_graph_with_edges.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(document_data, f, ensure_ascii=False, indent=4)

print(f"\nDone! JSON with edges: {output_json}")