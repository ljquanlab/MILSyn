from .head import FusionHead
from .model_utlis import Embeddings, \
    SmiCrossModalInteraction, MixGraphExtractor, MultimodalGatingNetwork, Cell_Self_Att, \
    FeatureFusion, ResCellCNN
import numpy as np
import torch
import torch.nn as nn


class MILSynNet(torch.nn.Module):

    def __init__(self,
                 hidden_dropout_prob=0.1,
                 feat_dim=64,
                 input_dim_drug=2586,
                 input_smi_drug=65,
                 output_dim=2048,
                 heads=8,
                 args=None):
        super(MILSynNet, self).__init__()
        self.args = args
        self.patch = 50
        self.max_smi_len = 179
        # view 1
        self.drug_emb = Embeddings(input_dim_drug, feat_dim, self.patch, 0.1)
        self.smi_emb = Embeddings(input_smi_drug, feat_dim, self.max_smi_len, 0.1)
        self.DDA = SmiCrossModalInteraction(embed_size=feat_dim, num_heads=heads, hidden_dim=feat_dim * 2)
        self.smi_DDA = SmiCrossModalInteraction(embed_size=feat_dim, num_heads=heads, hidden_dim=feat_dim * 2)
        self.cls_DDA = SmiCrossModalInteraction(embed_size=1536, num_heads=heads, hidden_dim=1536 * 2)

        self.cell_conv = ResCellCNN(in_channel=6, out_channel=feat_dim)
        self.cell_self_att = Cell_Self_Att(embed_size=feat_dim, num_heads=heads, hidden_dim=feat_dim * 2, layers=1)
        self.gate_fusion = MultimodalGatingNetwork(dim=feat_dim, output_dim=output_dim)
        # view 2
        self.gnn = MixGraphExtractor(output_dim=1024)
        # view 3
        self.llm_feat = FeatureFusion(feat_dim=1024)
        self.head = FusionHead()

    def forward(self, drugA, drugB, cell, graph, smiA, smiB, smiA_attention_mask, smiB_attention_mask,
                drugA_attention_mask, drugB_attention_mask, clsA, clsB, llm_cell, llmA, llmB):
        # view 1
        drugA = self.drug_emb(drugA).float()
        drugB = self.drug_emb(drugB).float()  # (bs,50,feat_dim)
        drugA, drugB, A2B, B2A = self.DDA(drugA, drugB, drugA_attention_mask, drugB_attention_mask)

        smiA = self.smi_emb(smiA.long()).float()
        smiB = self.smi_emb(smiB.long()).float()
        smiA, smiB, smiA2B, smiB2A = self.smi_DDA(smiA, smiB, smiA_attention_mask, smiB_attention_mask)

        clsA, clsB, _, _ = self.cls_DDA(clsA.unsqueeze(1), clsB.unsqueeze(1), None, None)

        cell = self.cell_conv(cell)  # torch.Size([32, 64, 170])
        cell, cell_att_matrix = self.cell_self_att(cell, None)
        f1 = self.gate_fusion(smiA, smiB, cell, drugA, drugB, clsA.squeeze(1), clsB.squeeze(1))
        # view 2
        f2 = self.gnn(graph)

        # view 3
        f3 = self.llm_feat(llm_cell, llmA, llmB)

        f = torch.cat([f1, f2, f3], dim=1)
        return self.head(f)

    def init_weights(self):
        self.head.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)