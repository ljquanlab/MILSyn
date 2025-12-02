from torch import nn
import math
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import GATv2Conv, RGCNConv, global_mean_pool as gap, global_max_pool as gmp, GCNConv
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        # 移除stride参数，固定为1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResCellCNN(nn.Module):
    def __init__(self, in_channel=6, out_channel=64, kernel = 15):
        super(ResCellCNN, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=kernel, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 使用MaxPool1d进行下采样
        self.res_blocks = nn.Sequential(
            ResidualBlock(16, 32, kernel_size=kernel),
            nn.MaxPool1d(kernel_size=2),  # 下采样2倍
            ResidualBlock(32, out_channel, kernel_size=kernel),
            nn.MaxPool1d(kernel_size=2),  # 下采样3倍
            ResidualBlock(out_channel, out_channel, kernel_size=kernel),
            nn.MaxPool1d(kernel_size=6),  # 下采样4倍
        )
        self.init_weights()
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        return x.transpose(1, 2)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.init_weights()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.lRelu = nn.ReLU()
        self.init_weights()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.lRelu(hidden_states)
        return hidden_states

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.init_weights()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class Cell_Self_Att(nn.Module):
    def __init__(self, embed_size=64, num_heads=8, hidden_dim=128, layers=1):
        super().__init__()
        self.diff_att = SelfAttention(embed_size, num_heads, 0.1)
        # self.diff_att = SHSA(dim=6, qk_dim=16, pdim=6) #(bs, 6, 64, 64)
        self.self_output = SelfOutput(embed_size, 0.1)
        self.norm = nn.LayerNorm(embed_size)
        self.layers = layers
        self.dense = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_size))
        self.init_weights()

    def forward(self, x, mask):
        # for i in range(self.layers):
        x = self.norm(x)
        x_att, cell_att_matrix = self.diff_att(x,mask)
        out = self.self_output(x_att, x)
        cellA_2 = x + out
        cellA_3 = self.norm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4
        return cellA_layer_output,cell_att_matrix

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class SmiCrossModalInteraction(nn.Module):
    def __init__(self, embed_size=128, num_heads=8, hidden_dim=256):
        super().__init__()
        self.att = CrossAttention(embed_size, num_heads, 0.1)
        self.norm = nn.LayerNorm(embed_size)
        self.self_output = SelfOutput(embed_size, 0.1)
        self.dense = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_size))
        self.init_weights()

    def forward(self, drug1_feats_norm, drug2_feats_norm, mask_a, mask_b):
        drug1_cross, A2B = self.att(drug1_feats_norm, drug2_feats_norm, mask_a)
        drug2_cross, B2A = self.att(drug2_feats_norm, drug1_feats_norm, mask_b)
        drug1_hidden = self.self_output(drug1_cross, drug1_feats_norm)
        drug2_hidden = self.self_output(drug2_cross, drug2_feats_norm)

        drugA_2 = drug1_feats_norm + drug1_hidden
        drugA_3 = self.norm(drugA_2)
        drugA_4 = self.dense(drugA_3)
        drugA_layer_output = drugA_2 + drugA_4

        drugB_2 = drug2_feats_norm + drug2_hidden
        drugB_3 = self.norm(drugB_2)
        drugB_4 = self.dense(drugB_3)
        drugB_layer_output = drugB_2 + drugB_4
        return drugA_layer_output, drugB_layer_output,A2B,B2A

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


# 边特征编码维度
EDGE_ATTR_DIM = 8  # (4种键类型 + 3种方向 + 1虚拟边标记)
class MixGraphExtractor(nn.Module):
    def __init__(self, num_features_xd=78, output_dim=1024, heads=8):
        super(MixGraphExtractor, self).__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_ATTR_DIM, 16),
            nn.ReLU(),
            # nn.LayerNorm(16)
        )
        self.rgcn_conv1 = RGCNConv(num_features_xd, num_features_xd, num_relations=5)
        self.rgcn_conv2 = RGCNConv(num_features_xd, num_features_xd * 2, num_relations=5)
        self.rgcn_conv3 = RGCNConv(num_features_xd * 2, num_features_xd * 4, num_relations=5)

        self.gat_conv1 = GATv2Conv(num_features_xd, num_features_xd, heads=2, edge_dim=16)
        self.gat_conv2 = GATv2Conv(num_features_xd * 2, num_features_xd * 2, heads=4, edge_dim=16)

        self.fc = nn.Sequential(
            nn.Linear(num_features_xd * 4 * 2 + num_features_xd * heads * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
        )
        self.init_weights()

    def forward(self, graph):
        x, edge_index, edge_attr, edge_type, batch = graph.x, graph.edge_index, graph.edge_attr, graph.edge_type, graph.batch

        x1 = self.rgcn_conv1(x, edge_index, edge_type)
        x1 = F.relu(x1)
        x1 = self.rgcn_conv2(x1, edge_index, edge_type)
        x1 = F.relu(x1)
        x1 = self.rgcn_conv3(x1, edge_index, edge_type)
        x1 = F.relu(x1)
        x1 = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        edge_attr = edge_attr.float()
        edge_embed = self.edge_encoder(edge_attr)
        x2 = self.gat_conv1(x, edge_index, edge_embed)
        x2 = F.relu(x2)
        x2 = self.gat_conv2(x2, edge_index, edge_embed)
        x2 = F.relu(x2)
        x2 = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class AttentionFusion(nn.Module):
    def __init__(self, embed_size=128, num_heads=8, hidden_dim=256, layers=1):
        super().__init__()
        self.diff_att = SelfAttention(embed_size, num_heads, 0.1)
        self.self_output = SelfOutput(embed_size, 0.1)
        self.intermediate = Intermediate(embed_size, hidden_dim)
        self.output = Output(hidden_dim, embed_size, 0.1)
        self.layers = layers

    def forward(self, x, mask):
        x_att, _ = self.diff_att(x, mask)
        out = self.self_output(x_att, x)
        intermediate_output = self.intermediate(out)
        layer_output = self.output(intermediate_output, out)
        return layer_output
        
class FeatureFusion(nn.Module):
    def __init__(self,feat_dim=1024, llm_dim = 1536):
        super(FeatureFusion, self).__init__()
        self.llm_feat = nn.Sequential(
            nn.Linear(llm_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        self.cell_feat = nn.Sequential(
            nn.Linear(llm_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        self.position_embeddings = nn.Embedding(3, feat_dim)
        self.fusion = AttentionFusion(embed_size=feat_dim, num_heads=8, hidden_dim=feat_dim * 2, layers=1)
        self.norm = nn.LayerNorm(feat_dim)
    def forward(self, llm_cell, llmA, llmB):
        device = llm_cell.device
        llm_cell = self.cell_feat(llm_cell)
        llmA = self.llm_feat(llmA)
        llmB = self.llm_feat(llmB)
        features = torch.stack([llm_cell, llmA, llmB], dim=1)

        position_ids = torch.arange(features.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(llmA.size(0),-1)
        position_embeddings = self.position_embeddings(position_ids)
        f = self.norm(features+position_embeddings)
        return self.fusion(f,None).view(f.size(0), -1)

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class MultimodalGatingNetwork(nn.Module):
    def __init__(self, dim=32, output_dim=1024, cls_dim = 1536):
        super(MultimodalGatingNetwork, self).__init__()
        self.gated_fp = GLU(dim, dim)  # bs,50,dim
        self.gated_smi = GLU(dim, dim)  # bs,179,dim
        self.gated_cell = GLU(dim, dim)
        self.gated_cls = GLU(cls_dim,cls_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.fp_mlp = nn.Sequential(
            nn.Linear(50 * dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.smi_mlp = nn.Sequential(
            nn.Linear(179 * dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(169 * dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.cls_mlp = nn.Sequential(
            nn.Linear(cls_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # self.norm = nn.LayerNorm(cls_dim)
        self.init_weights()

    def forward(self, smi1,  smi2, cell, fp1, fp2, clsA, clsB):
        batch_size = smi1.shape[0]
        fp1 = self.gated_fp(fp1)
        fp2 = self.gated_fp(fp2)
        smi1 = self.gated_smi(smi1)
        smi2 = self.gated_smi(smi2)
        cell = self.gated_cell(cell)
        # clsA = self.norm(clsA)
        # clsB = self.norm(clsB)
        clsA = self.gated_cls(clsA)
        clsB = self.gated_cls(clsB)

        fp1 = self.fp_mlp(fp1.view(batch_size, -1))
        fp2 = self.fp_mlp(fp2.view(batch_size, -1))
        smi1 = self.smi_mlp(smi1.view(batch_size, -1))
        smi2 = self.smi_mlp(smi2.view(batch_size, -1))
        cell = self.cell_mlp(cell.view(batch_size, -1))
        clsA = self.cls_mlp(clsA)
        clsB = self.cls_mlp(clsB)

        fp_cell1 = fp1 * cell
        smi_cell1 = smi1 * cell
        cls_cell1 = clsA * cell
        fp_cell2 = fp2 * cell
        smi_cell2 = smi2 * cell
        cls_cell2 = clsB * cell

        f = self.relu(torch.cat([ fp_cell1, smi_cell1, cls_cell1, fp_cell2, smi_cell2, cls_cell2], dim=-1))
        return f

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_0


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, drugA, drugB, drugA_attention_mask):
        # update drugA
        mixed_query_layer = self.query(drugA)
        mixed_key_layer = self.key(drugB)
        mixed_value_layer = self.value(drugB)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if drugA_attention_mask == None:
            attention_scores = attention_scores
        else:
            attention_scores = attention_scores + drugA_attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs_0
