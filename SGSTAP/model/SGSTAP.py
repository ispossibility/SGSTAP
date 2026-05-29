import math

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json

import sys
import os

from torch.nn import Parameter

from lib.utils import build_fine_to_coarse_mapping

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, out_dim=1):
        super().__init__()
        self.pred_head = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=1
        )

    def forward(self, x):
        return self.pred_head(x)

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, day_len=24, week_len=7):
        super().__init__()
        self.d_model = d_model
        self.day_len = day_len
        self.week_len = week_len
        pe_abs = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe_abs[:, 0::2] = torch.sin(position * div_term)
        pe_abs[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_abs", pe_abs.unsqueeze(0))
        pe_day = torch.zeros(day_len, d_model)
        pos_day = torch.arange(0, day_len).unsqueeze(1)
        pe_day[:, 0::2] = torch.sin(pos_day * div_term)
        pe_day[:, 1::2] = torch.cos(pos_day * div_term)
        self.register_buffer("pe_day", pe_day)  # (24, C)
        pe_week = torch.zeros(week_len, d_model)
        pos_week = torch.arange(0, week_len).unsqueeze(1)
        pe_week[:, 0::2] = torch.sin(pos_week * div_term)
        pe_week[:, 1::2] = torch.cos(pos_week * div_term)
        self.register_buffer("pe_week", pe_week)  # (7, C)
        self.alpha_abs = nn.Parameter(torch.tensor(0.1))
        self.alpha_day = nn.Parameter(torch.tensor(0.1))
        self.alpha_week = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, T, C = x.shape
        device = x.device
        pe_abs = self.pe_abs[:, :T].to(device)
        t = torch.arange(T, device=device)
        day_idx = t % self.day_len
        pe_day = self.pe_day[day_idx].unsqueeze(0)
        week_idx = torch.div(t, self.day_len, rounding_mode='floor') % self.week_len
        pe_week = self.pe_week[week_idx].unsqueeze(0)
        return (
            x
            + self.alpha_abs * pe_abs
            + self.alpha_day * pe_day
            + self.alpha_week * pe_week
        )

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()

        self.pos_emb = TemporalPositionalEncoding(
            d_model=hidden_dim,
            max_len=500,
            day_len=24,
            week_len=7
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, x):
        x = self.pos_emb(x)
        x = self.encoder(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.transformer = TemporalTransformer(
            hidden_dim=hidden_dim,
            num_heads=4,
            ff_dim=4 * hidden_dim
        )

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.transformer(x)
        return x

class HGNNConv(nn.Module):
    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.zeros(out_ft))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, H):
        N, E = H.shape
        Dv = torch.diag(H.sum(dim=1).float() + 1e-6)
        De = torch.diag(H.sum(dim=0).float() + 1e-6)
        Dv_inv_sqrt = torch.linalg.inv(torch.sqrt(Dv))
        De_inv = torch.linalg.inv(De)
        L = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt
        X = X @ self.weight + self.bias
        X = L @ X
        return X


class DynamicHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, hypergraphs):
        super().__init__()
        self.hgnn = HGNNConv(in_dim, hidden_dim)
        self.hypergraphs = nn.ParameterDict({
            str(k): nn.Parameter(v, requires_grad=False)
            for k, v in hypergraphs.items()
        })

    def forward(self, graph_feature):
        B, T, N, D = graph_feature.shape
        device = graph_feature.device
        weather_onehot = graph_feature[..., 9:14]
        weather_ratio = weather_onehot.mean(dim=2)
        weather_idx = weather_ratio.argmax(dim=-1)
        out = torch.zeros(B, T, N, self.hgnn.weight.shape[1], device=device)
        for b in range(B):
            for t in range(T):
                w = int(weather_idx[b, t].item())
                H = self.hypergraphs[str(w)]
                X = graph_feature[b, t]
                out[b, t] = self.hgnn(X, H)
        return out



class CMModule(nn.Module):
    def __init__(self, grid_in_channel,spatial_hidden):
        super(CMModule, self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=spatial_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=spatial_hidden, out_channels=spatial_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.course_grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=spatial_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=spatial_hidden, out_channels=spatial_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, grid_input, course_grid_input):
        batch_size, T, D, W, H = grid_input.shape
        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)
        grid_output = conv_output.view(batch_size, T, -1, W, H)
        batch_size, T, D, course_W, course_H = course_grid_input.shape
        course_grid_input = course_grid_input.view(-1, D, course_W, course_H)
        course_conv_output = self.course_grid_conv(course_grid_input)
        course_grid_output = course_conv_output.view(batch_size, T, -1, course_W, course_H)
        return grid_output, course_grid_output

class HCGModule(nn.Module):
    def __init__(self, num_of_graph_feature,  hgcn_hidden, north_south_map, west_east_map, Hypergraphs, course_hypergraphs):
        super(HCGModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.hgnn = DynamicHGNN(num_of_graph_feature, hgcn_hidden, Hypergraphs)
        self.course_hgnn = DynamicHGNN(num_of_graph_feature, hgcn_hidden, course_hypergraphs)

    def forward(self, graph_feature, course_graph_feature):
        batch_size, T, D1, N = graph_feature.shape
        graph_feature = graph_feature.permute(0, 1, 3, 2).contiguous()
        course_graph_feature = course_graph_feature.permute(0, 1, 3, 2).contiguous()
        graph_feature_hg = self.hgnn(graph_feature)
        course_graph_feature_hg = self.course_hgnn(course_graph_feature)
        graph_output = graph_feature_hg.permute(0,1,3,2).view(batch_size,T,-1,self.north_south_map, self.west_east_map)
        course_graph_output = course_graph_feature_hg.permute(0,1,3,2).view(batch_size,T,-1,9, 9)
        return graph_output, course_graph_output


class SGSTAP(nn.Module):
    def __init__(self, grid_in_channel,
                 spatial_hidden,
                 num_of_graph_feature,
                 hgcn_hidden,
                 temporal_hidden,
                 north_south_map, west_east_map,Hypergraphs,course_hypergraphs):
        super(SGSTAP, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map

        self.cm_module = CMModule(grid_in_channel,spatial_hidden)
        self.hcg_module = HCGModule(num_of_graph_feature, hgcn_hidden, north_south_map, west_east_map, Hypergraphs, course_hypergraphs)
        self.grid_weigth = nn.Conv2d(in_channels=spatial_hidden, out_channels=temporal_hidden, kernel_size=1)
        self.graph_weigth = nn.Conv2d(in_channels=hgcn_hidden, out_channels=temporal_hidden, kernel_size=1)
        self.course_grid_weigth = nn.Conv2d(in_channels=spatial_hidden, out_channels=temporal_hidden, kernel_size=1)
        self.course_graph_weigth = nn.Conv2d(in_channels=hgcn_hidden, out_channels=temporal_hidden, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer(
            "M_9x9",
            torch.from_numpy(
                build_fine_to_coarse_mapping(
                    W_fine=20, H_fine=20, block_size=4, stride=2
                )
            ).float()
        )
        self.coarse_proj = nn.Conv2d(temporal_hidden, temporal_hidden, kernel_size=1)
        # self.fusion_proj = nn.Conv2d(2 * temporal_hidden, temporal_hidden, kernel_size=1)
        self.temporal_model = TemporalModel(temporal_hidden)
        self.pred_head = PredictionHead(temporal_hidden, out_dim=1)

    def forward(self, grid_input, course_grid_input, graph_feature, course_graph_feature):
        grid_input = grid_input.float()
        course_grid_input = course_grid_input.float()
        graph_feature = graph_feature.float()
        course_graph_feature = course_graph_feature.float()
        batch_size, T, D, W, H = grid_input.shape
        batch_size, _, _, course_W, course_H = course_grid_input.shape
        grid_output, course_grid_output = self.cm_module(grid_input, course_grid_input)
        graph_output, course_graph_output = self.hcg_module(graph_feature, course_graph_feature)
        grid_output = self.grid_weigth(grid_output.view(batch_size*T,-1,W,H))
        graph_output = self.graph_weigth(graph_output.view(batch_size*T,-1,W,H))
        gate = torch.sigmoid(grid_output + graph_output)
        fused = gate * grid_output + (1 - gate) * graph_output
        course_grid_output = self.course_grid_weigth(course_grid_output.view(batch_size*T,-1,course_W,course_H))
        course_graph_output = self.course_graph_weigth(course_graph_output.view(batch_size*T,-1,course_W,course_H))
        course_gate = torch.sigmoid(course_grid_output + course_graph_output)
        course_fused = course_gate * course_grid_output + (1 - course_gate) * course_graph_output
        _, c, Wc, Hc = course_fused.shape
        course_flat = course_fused.view(batch_size*T, c, Wc * Hc)
        fine_flat = torch.matmul(course_flat, self.M_9x9.T)  # (B*T, C, 400)
        fine_from_coarse = fine_flat.view(batch_size*T, c, W, H)
        fine_from_coarse = self.coarse_proj(fine_from_coarse)
        fused_total = fused + self.alpha*fine_from_coarse
        fused_total = fused_total.view(batch_size,T,-1,W,H)
        fused_total = fused_total.permute(0,3,4,1,2).reshape(batch_size*W*H,T,-1);
        fusion_out = self.temporal_model(fused_total)
        fusion_out = fusion_out.mean(dim=1)
        fusion_out = fusion_out.view(batch_size,W,H,-1).permute(0,3,1,2)
        final_output = self.pred_head(fusion_out)
        return final_output
