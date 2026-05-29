import numpy as np
import pandas as pd
import torch
import sys
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, jaccard
from numpy.linalg import norm
from scipy.stats import pearsonr
from tqdm import tqdm
from collections import deque

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np
import torch
import torch.nn.functional as F

def split_time_by_weather(all_data):
    T = all_data.shape[0]
    data = torch.from_numpy(all_data)
    weather_time_idx = {}
    for w in range(5):
        weather_flag = data[:, 41 + w]
        weather_flag = weather_flag.mean(dim=(1, 2))
        idx = torch.where(weather_flag > 0.5)[0]
        weather_time_idx[w] = idx
    return weather_time_idx

def build_node_feature_by_weather(all_data, time_idx):
    data = torch.from_numpy(all_data).float()
    T, D, W, H = data.shape
    N = W * H
    poi = data[4321, 33:40]
    poi = poi.permute(1, 2, 0).reshape(N, 7)
    inflow = data[time_idx, 46].reshape(len(time_idx), -1)
    outflow = data[time_idx, 47].reshape(len(time_idx), -1)
    flow_feat = torch.stack([
        inflow.mean(dim=0),
        outflow.mean(dim=0),
        inflow.std(dim=0),
        outflow.std(dim=0)
    ], dim=1)
    node_feat = torch.cat([poi, flow_feat], dim=1)
    node_feat = F.normalize(node_feat, dim=1)

    return node_feat

def build_hypergraph_once(node_feat, k=30, sim_threshold=0.8):
    sim = node_feat @ node_feat.T
    N = sim.size(0)
    hyperedges = []
    for i in range(N):
        scores = sim[i]
        topk = torch.topk(scores, k + 1).indices
        topk = topk[scores[topk] > sim_threshold]
        if topk.numel() >= 2:
            hyperedges.append(topk)
    if len(hyperedges) == 0:
        return torch.zeros((N, 1))
    H = torch.zeros((N, len(hyperedges)))
    for e, nodes in enumerate(hyperedges):
        H[nodes, e] = 1.0
    return H


def build_weather_hypergraphs(all_data, k=30, sim_threshold=0.8):
    weather_time_idx = split_time_by_weather(all_data)
    H_dict = {}
    for w in range(5):
        idx = weather_time_idx[w]
        if len(idx) < 10:
            print(f"Weather {w}: not enough samples, skip")
            continue
        node_feat = build_node_feature_by_weather(all_data, idx)
        H = build_hypergraph_once(
            node_feat,
            k=k,
            sim_threshold=sim_threshold
        )
        H_dict[w] = H
    return H_dict

def build_fine_to_coarse_mapping(
        W_fine,
        H_fine,
        block_size,
        stride
):
    Wc = (W_fine - block_size) // stride + 1
    Hc = (H_fine - block_size) // stride + 1
    num_fine = W_fine * H_fine
    num_coarse = Wc * Hc
    M = np.zeros((num_fine, num_coarse), dtype=np.float32)
    for i in range(Wc):
        for j in range(Hc):
            coarse_id = i * Hc + j
            w0, h0 = i * stride, j * stride
            fine_ids = []
            for wi in range(w0, w0 + block_size):
                for hj in range(h0, h0 + block_size):
                    fine_id = wi * H_fine + hj
                    fine_ids.append(fine_id)
            weight = 1.0 / len(fine_ids)
            for fid in fine_ids:
                M[fid, coarse_id] += weight
    return M

def aggregate_grid_node_map(grid_node_map, W, H, block_size, stride, weight_matrix=None):

    N = grid_node_map.shape[1]
    W_new = (W - block_size) // stride + 1
    H_new = (H - block_size) // stride + 1
    grid_node_map_2d = grid_node_map.reshape(W, H, N)
    grid_node_map_new = np.zeros((W_new, H_new, N))
    for i in range(W_new):
        for j in range(H_new):
            w0, h0 = i * stride, j * stride
            block = grid_node_map_2d[w0:w0 + block_size, h0:h0 + block_size, :]
            block = block.reshape(-1, N)
            if weight_matrix is not None:
                weights = weight_matrix[w0:w0 + block_size, h0:h0 + block_size].flatten()[:, None]
                grid_node_map_new[i, j, :] = (block * weights).sum(axis=0)
            else:
                grid_node_map_new[i, j, :] = block.mean(axis=0)
    grid_node_map_new = grid_node_map_new.reshape(W_new * H_new, N)
    return grid_node_map_new

class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33:40] = (data[:, 33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:, 46] = (data[:, 46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:, 47] = (data[:, 47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)         D 是特征维度
        """
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))  # (T*W*H,D)
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33] = (data[:, 33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:, 39] = (data[:, 39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


def mask_loss(predicts, labels, region_mask, data_type="nyc"):
    batch_size, pre_len, _, _ = predicts.shape

    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()

    loss = ((labels - predicts) * region_mask) ** 2

    if data_type == 'nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type == 'chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 1 / 17)
        index_3 = (labels > 1 / 17) & (labels <= 2 / 17)
        index_4 = labels > 2 / 17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return torch.mean(loss)


@torch.no_grad()
def compute_loss(net, dataloader, risk_mask,
                 global_step, device,
                 data_type='nyc'):
    net.eval()
    temp = []
    for feature, course_feature, graph_feature, course_graph_feature, label in dataloader:
        feature, course_feature, graph_feature, course_graph_feature, label = feature.to(device), course_feature.to(
            device), graph_feature.to(device), course_graph_feature.to(device), label.to(device)

        l = mask_loss(net(feature, course_feature, graph_feature, course_graph_feature), label, risk_mask,
                      data_type)
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask,
                         global_step, scaler, device):
    net.eval()
    prediction_list = []
    label_list = []
    for feature, course_feature, graph_feature, course_graph_feature, label in dataloader:
        feature, course_feature, graph_feature, course_graph_feature, label = feature.to(device), course_feature.to(
            device), graph_feature.to(device), course_graph_feature.to(device), label.to(device)
        prediction_list.append(net(feature, course_feature, graph_feature, course_graph_feature).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)

    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)

    rmse_, recall_, map_, mae, mape, r_2 = mask_evaluation_np(inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, mae, mape, r_2, inverse_trans_pre, inverse_trans_label