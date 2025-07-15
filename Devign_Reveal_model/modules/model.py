import torch
from dgl.nn import GatedGraphConv
import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import RobertaModel, RobertaTokenizer
import pickle
from sklearn.model_selection import train_test_split


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        print('input_dim')
        print(input_dim)
        print('output_dim')
        print(output_dim)
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(self.concat_dim, self.out_dim)


    def forward(self, batch, device, return_embedding=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=True, device=device)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)

        Y_1 = self.maxpool1(f.relu(self.conv_l1(h_i.transpose(1, 2))))
        Y_2 = self.maxpool2(f.relu(self.conv_l2(Y_1))).transpose(1, 2)

        Z_1 = self.maxpool1_for_concat(f.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(f.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)

        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)  # [batch_size, 1]

        # if return_embedding:
        #     return Z_2.mean(dim=1)  # shape = [batch_size, output_dim] (e.g., 64 or 128)
        #     # return avg  # Đây là vector đầu ra cần cho mô hình kết hợp

        if return_embedding:
            graph_repr = Z_2.mean(dim=1)  # [batch_size, concat_dim]
            return self.proj(graph_repr)  # [batch_size, output_dim]


        result = self.sigmoid(avg).squeeze(dim=-1)
        return result  # Trường hợp dùng Devign bình thường (không kết hợp)



class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, device):
        self.ggnn = self.ggnn.to(device)
        self.classifier = self.classifier.to(device)
        graph, features, edge_types = batch.get_network_inputs(cuda=True, device=device)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result





# === Multimodal Model Components ===

class SequenceModel(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", output_dim=768):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.encoder = RobertaModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
        self.linear = nn.Linear(self.encoder.config.hidden_size, output_dim)

    def forward(self, func_list):
        inputs = self.tokenizer(
            func_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.encoder.device)

        with torch.no_grad():  # freeze CodeBERT
            outputs = self.encoder(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.linear(cls_output)

class ConcatFusion(nn.Module):
    def __init__(self, d1, d2, out_dim):
        super().__init__()
        self.fc = nn.Linear(d1 + d2, out_dim)
    def forward(self, x1, x2):
        return self.fc(torch.cat((x1, x2), dim=1))

class GMUFusion(nn.Module):
    def __init__(self, d1, d2, out_dim):
        super().__init__()
        self.gate = nn.Linear(d1 + d2, out_dim)
        self.fc1 = nn.Linear(d1, out_dim)
        self.fc2 = nn.Linear(d2, out_dim)
    def forward(self, x1, x2):
        print("x1:", x1.shape, "x2:", x2.shape)
        z = torch.sigmoid(self.gate(torch.cat((x1, x2), dim=1)))
        h1 = torch.tanh(self.fc1(x1))
        h2 = torch.tanh(self.fc2(x2))
        return z * h1 + (1 - z) * h2

class CrossAttentionFusion(nn.Module):
    def __init__(self, d1, d2, out_dim):
        super().__init__()
        self.query = nn.Linear(d1, out_dim)
        self.key = nn.Linear(d2, out_dim)
        self.value = nn.Linear(d2, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
    def forward(self, x1, x2):
        Q = self.query(x1).unsqueeze(1)
        K = self.key(x2).unsqueeze(1)
        V = self.value(x2).unsqueeze(1)
        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) / (Q.size(-1)**0.5), dim=-1)
        return self.out_proj((attn_weights @ V).squeeze(1))

class CombinedModel(nn.Module):
    def __init__(self, fusion_type='gmu', graph_dim=128, seq_dim=768, fusion_dim=128, device='cuda:0', feature_size=100, max_edge_type=5):
        super().__init__()
        self.device = device
        self.graph_model = DevignModel(
            input_dim=feature_size,
            output_dim=graph_dim,
            # output_dim=64,  # <-- hoặc 128
            max_edge_types=max_edge_type,
            num_steps=8
        )
        self.seq_model = SequenceModel(output_dim=seq_dim)

        print('graph_dim')
        print(graph_dim)
        print('seq_dim')
        print(seq_dim)
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(graph_dim, seq_dim, fusion_dim)
        elif fusion_type == 'gmu':
            # self.fusion = GMUFusion(graph_dim, seq_dim, fusion_dim)
            self.fusion = GMUFusion(d1=graph_dim, d2=seq_dim, out_dim=fusion_dim)

        elif fusion_type == 'crossattn':
            self.fusion = CrossAttentionFusion(graph_dim, seq_dim, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion: {fusion_type}")

        self.classifier = nn.Linear(fusion_dim, 2)

    def forward(self, batch):
        graph_out = self.graph_model(batch, device=self.device, return_embedding=True)  # Lấy vector
        func_list = batch.func_list  # Đã được gán sẵn
        seq_out = self.seq_model(func_list)  # Vector từ CodeBERT
        print(f"graph_out.shape = {graph_out.shape}, seq_out.shape = {seq_out.shape}")
        fused = self.fusion(graph_out, seq_out)
        return self.classifier(fused)




