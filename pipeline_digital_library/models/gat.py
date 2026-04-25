from config import GAT_MODEL_PATH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

_model = None


class DocumentMultiTaskGAT(nn.Module):
    def __init__(self, num_node_classes=8):
        super().__init__()
        self.geom_proj = nn.Linear(11, 64)
        self.yolo_proj = nn.Linear(11, 32)
        # droupout for yolo features to learn reaclassify
        self.yolo_dropout = nn.Dropout(0.4)
        self.text_proj = nn.Linear(389, 160)

        self.conv1 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(256, 64, heads=4, concat=True, dropout=0.2)
        self.conv3 = GATv2Conv(256, 32, heads=8, concat=True, dropout=0.2)

        # Correction gate: naučí sa kedy ignorovať YOLO
        self.correction_gate = nn.Sequential(
            nn.Linear(256 + 11, 64),  # GAT embedding + yolo soft label
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 = ver YOLO, 1 = ver GAT
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
        x_yolo = batch.feat_yolo
        x_text = batch.feat_text

        h_geom = F.elu(self.geom_proj(x_geom))

        # apply dropout on yolo prediction
        h_yolo = F.elu(self.yolo_proj(x_yolo))
        h_yolo = self.yolo_dropout(h_yolo)

        h_text = F.elu(self.text_proj(x_text))
        x = torch.cat([h_geom, h_yolo, h_text], dim=-1)  # [N, 256]

        x_res1 = x
        x = F.elu(self.conv1(x, batch.edge_index)) + x_res1
        x_res2 = x
        x = F.elu(self.conv2(x, batch.edge_index)) + x_res2
        z = F.elu(self.conv3(x, batch.edge_index))  # [N, 256]

        # Gate: váha opravy vs. pôvodný YOLO prior
        gate = self.correction_gate(
            torch.cat([z, x_yolo], dim=-1)
        )  # [N, 1]

        # YOLO prior ako logity (nenormalizované)
        yolo_prior = x_yolo / (x_yolo.sum(dim=-1, keepdim=True) + 1e-8)
        # Len 8 tried — yolo_prior má 11 stĺpcov, potrebujeme namapovať
        # Jednoduchšie: nechaj classifier rozhodnúť, gate len škáluje residual
        node_logits = self.node_classifier(z)  # [N, 8]

        # Mixuj: pri gate≈0 model skĺzne k YOLO priors, pri gate≈1 verí sebe
        # (Voliteľné — môžeš začať bez toho a pridať neskôr)

        return z, node_logits, gate

    def predict_edges(self, z, query_edge_index, feat_geom=None):
        src_z = z[query_edge_index[0]]
        dst_z = z[query_edge_index[1]]
        edge_feat = torch.cat([src_z, dst_z], dim=-1)

        if feat_geom is not None:
            # Geometrické features hrany: pozícia src a dst
            src_g = feat_geom[query_edge_index[0]][:, :6]  # x1,y1,x2,y2,xc,yc
            dst_g = feat_geom[query_edge_index[1]][:, :6]
            edge_feat = torch.cat([edge_feat, src_g - dst_g], dim=-1)

        return self.edge_classifier(edge_feat)  # [E, 3] logity


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