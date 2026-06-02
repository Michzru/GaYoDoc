from config import GAT_MODEL_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


DOCLING_TO_GAT = torch.tensor(
    # 0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16
    [  0, 7, 3, 7, 5, 6, 1, 4, 2, 7,  7,  2,  7,  7,  7,  7,  7],
    dtype=torch.long
)

_model = None


class DocumentMultiTaskGAT(nn.Module):
    def __init__(self, num_node_classes=8):
        super().__init__()
        self.geom_proj = nn.Linear(11, 64)

        # docling má 17 tried namiesto 11
        self.docling_proj = nn.Linear(17, 32)
        self.docling_dropout = nn.Dropout(0.2)

        # 13 manual features + 384 MiniLM
        self.text_proj = nn.Linear(397, 160)

        # 64 + 32 + 160 = 256 — rovnaké ako predtým
        self.conv1 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(256, 32, heads=8, concat=True, dropout=0.2)

        # correction gate: 256 (z) + 17 (docling soft-label)
        self.correction_gate = nn.Sequential(
            nn.Linear(256 + 17, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_node_classes)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(512 + 6, 128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, batch):
        x_geom = batch.feat_geom
        x_docling = batch.feat_docling
        x_text = batch.feat_text

        h_geom = F.elu(self.geom_proj(x_geom))
        h_docling = F.elu(self.docling_proj(x_docling))
        h_docling = self.docling_dropout(h_docling)
        h_text = F.elu(self.text_proj(x_text))

        x = torch.cat([h_geom, h_docling, h_text], dim=-1)
        x_res1 = x
        x = F.elu(self.conv1(x, batch.edge_index)) + x_res1
        x_res2 = x
        x = F.elu(self.conv2(x, batch.edge_index)) + x_res2
        z = F.elu(self.conv3(x, batch.edge_index))

        gate = self.correction_gate(
            torch.cat([z, x_docling], dim=-1)
        )  # [N, 1]

        gat_logits = self.node_classifier(z)  # [N, 8]

        # Docling soft-label → 8-dim GAT priestor
        mapping = DOCLING_TO_GAT.to(x_docling.device)
        docling_8 = torch.zeros(x_docling.shape[0], 8, device=x_docling.device)
        for docling_idx in range(17):
            gat_idx = mapping[docling_idx]
            docling_8[:, gat_idx] = docling_8[:, gat_idx] + x_docling[:, docling_idx]

        # gate rozhoduje slobodne — bez blokovania
        node_logits = gate * gat_logits + (1.0 - gate) * docling_8

        return z, node_logits, gate

    def predict_edges(self, z, query_edge_index, feat_geom=None):
        src_z = z[query_edge_index[0]]
        dst_z = z[query_edge_index[1]]
        edge_feat = torch.cat([src_z, dst_z], dim=-1)

        if feat_geom is not None:
            src_g = feat_geom[query_edge_index[0]][:, :6]
            dst_g = feat_geom[query_edge_index[1]][:, :6]
            edge_feat = torch.cat([edge_feat, src_g - dst_g], dim=-1)

        return self.edge_classifier(edge_feat)  # [E, 3]


def get_gat_model(verbose, gpu = True):
    global _model
    if _model is None:
        if gpu:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

        _model = DocumentMultiTaskGAT(num_node_classes=8).to(device)
        _model.load_state_dict(torch.load(GAT_MODEL_PATH, map_location=device))
        _model.eval()
        if verbose:
            print(f"GAT model loaded on {device}.")
    return _model