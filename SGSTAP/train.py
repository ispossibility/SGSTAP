import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import json
import configparser
import pickle as pkl
from time import time
from datetime import datetime
import shutil
import argparse
import random
import math

import sys
import os

from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.dataloader import normal_and_generate_dataset_time, get_mask, get_adjacent, get_grid_node_map_maxtrix
from lib.early_stop import EarlyStopping
from model.SGSTAP import SGSTAP
from lib.utils import mask_loss, compute_loss, predict_and_evaluate, aggregate_grid_node_map, build_weather_hypergraphs

# 设置默认配置
DEFAULT_CONFIG = 'config/nyc/SGSTAP_NYC_Config.json'
DEFAULT_GPUS = '1'



parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--test", action="store_true", help="test program")

args = parser.parse_args()
# config_filename = args.config
with open(DEFAULT_CONFIG, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = DEFAULT_GPUS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

north_south_map = config['north_south_map']
west_east_map = config['west_east_map']

all_data_filename = config['all_data_filename']
course_data_filename = config['aggregated']
mask_filename = config['mask_filename']


patience = config['patience']
delta = config['delta']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

train_rate = config['train_rate']
valid_rate = config['valid_rate']

recent_prior = config['recent_prior']
week_prior = config['week_prior']
one_day_period = config['one_day_period']
days_of_week = config['days_of_week']
pre_len = config['pre_len']
seq_len = recent_prior + week_prior
# seq_len = recent_prior

training_epoch = config['training_epoch']


def training(net,
             training_epoch,
             train_loader,
             val_loader,
             test_loader,
             risk_mask,
             trainer,
             early_stop,
             device,
             scaler,
             data_type='nyc'
             ):
    global_step = 1
    for epoch in tqdm(range(1, training_epoch + 1)):
        net.train()
        batch = 1
        temp = []
        start_time = time()
        for train_feature, course_train_feature, gragh_feature, course_graph_feature, train_label in train_loader:
            train_feature, course_train_feature, gragh_feature, course_graph_feature, train_label = train_feature.to(
                device), course_train_feature.to(device), gragh_feature.to(device), course_graph_feature.to(
                device), train_label.to(device)
            l = mask_loss(net(train_feature, course_train_feature, gragh_feature, course_graph_feature), train_label,
                          risk_mask, data_type=data_type)
            temp.append(l.cpu().item())
            trainer.zero_grad()
            l.backward()
            trainer.step()
            if batch % 100 == 0:
                print(f"Epoch {epoch}, batch {batch}")
            batch += 1
            global_step += 1
        loss_mean = sum(temp) / len(temp)
        print('global step: %s, epoch: %s,train loss：%.6f, time: %.2fs' % (
        global_step - 1, epoch, loss_mean, time() - start_time), flush=True)
        val_loss = compute_loss(net, val_loader, risk_mask, global_step - 1, device, data_type)
        print('global step: %s, epoch: %s,val loss：%.6f' % (global_step - 1, epoch, val_loss), flush=True)
        if epoch == 1 or val_loss < early_stop.best_score:
            test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2, test_inverse_trans_pre, test_inverse_trans_label = \
                predict_and_evaluate(net, test_loader, risk_mask, global_step - 1, scaler, device)
            print(
                'global step: %s, epoch: %s, test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,test MAE: %.4f,test MAPE: %.4f,test R_2: %.4f'
                % (
                    global_step - 1, epoch, test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2),
                flush=True)
        early_stop(val_loss, test_rmse, test_recall, test_map, test_mae, test_mape, test_r_2,
                   test_inverse_trans_pre, test_inverse_trans_label)

        if early_stop.early_stop:
            print("Early Stopping in global step: %s, epoch: %s" % (global_step, epoch), flush=True)
            print(
                'best test RMSE: %.4f,best test Recall: %.2f%%,best test MAP: %.4f, best test MAE: %.4f, best test MAPE: %.4f, best test R_2: %.4f,'
                % (early_stop.best_rmse, early_stop.best_recall, early_stop.best_map, early_stop.best_mae,
                   early_stop.best_mape, early_stop.best_r_2,), flush=True)
            break
    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map, early_stop.best_mae, early_stop.best_mape, early_stop.best_r_2

def main(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    spatial_hidden = config['spatial_hidden']
    hgcn_hidden = config['hgcn_hidden']
    temporal_hidden = config['temporal_hidden']

    loaders = []
    scaler = ""
    train_data_shape = ""
    graph_feature_shape = ""

    for idx, (x, y, course_x, course_y, scaler) in enumerate(normal_and_generate_dataset_time(
            all_data_filename,
            course_data_filename,
            train_rate=train_rate,
            valid_rate=valid_rate,
            recent_prior=recent_prior,
            week_prior=week_prior,
            one_day_period=one_day_period,
            days_of_week=days_of_week,
            pre_len=pre_len)):
        if 'nyc' in all_data_filename:
            graph_feat_idx = [0] + list(range(33, 48))
            graph_x = x[:, :, graph_feat_idx, :, :].reshape(
                (x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            course_graph_x = course_x[:, :, graph_feat_idx, :, :].reshape(
                (course_x.shape[0], course_x.shape[1], -1, 9 * 9))
        if 'chicago' in all_data_filename:
            graph_x = x[:, :, [0, 39, 40], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            course_graph_x = course_x[:, :, [0, 39, 40], :, :].reshape(
                (course_x.shape[0], course_x.shape[1], -1, 9 * 9))
        print("feature:", str(x.shape), "label:", str(y.shape))
        print("graph_x:", str(graph_x.shape))
        print("course_graph_x:", str(course_graph_x.shape), "course_y:", str(course_y.shape))
        if idx == 0:
            scaler = scaler
            train_data_shape = x.shape
            course_train_data_shape = course_x.shape
            graph_feature_shape = graph_x.shape
            course_graph_feature_shape = course_graph_x.shape
            loaders.append(Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(x),
                    torch.from_numpy(course_x),
                    torch.from_numpy(graph_x),
                    torch.from_numpy(course_graph_x),
                    torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))
        elif idx == 1:
            loaders.append(Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(x),
                    torch.from_numpy(course_x),
                    torch.from_numpy(graph_x),
                    torch.from_numpy(course_graph_x),
                    torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))
        elif idx == 2:
            loaders.append(Data.DataLoader(
                Data.TensorDataset(
                    torch.from_numpy(x),
                    torch.from_numpy(course_x),
                    torch.from_numpy(graph_x),
                    torch.from_numpy(course_graph_x),
                    torch.from_numpy(y)
                ),
                batch_size=batch_size,
                shuffle=(idx == 0)
            ))

    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    spatial_hidden = config['spatial_hidden']
    hgcn_hidden = config['hgcn_hidden']
    temporal_hidden = config['temporal_hidden']
    print("train_data_shape[2]:", str(train_data_shape[2]))
    print("course_train_data_shape[2]:", str(course_train_data_shape[2]))
    print("num_of_gru_layers:", str(spatial_hidden))
    print("seq_len:", str(seq_len))
    print("pre_len:", str(pre_len))
    print("graph_feature_shape[2]:", str(graph_feature_shape[2]))
    print("course_graph_feature_shape[2]:", str(course_graph_feature_shape[2]))
    all_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)
    H_dict = build_weather_hypergraphs(all_data)
    course_data = pkl.load(open(course_data_filename, 'rb')).astype(np.float32)
    course_H_dict = build_weather_hypergraphs(course_data)
    SGSTAP_Model = SGSTAP(train_data_shape[2], spatial_hidden,
                        graph_feature_shape[2], hgcn_hidden,
                        temporal_hidden, north_south_map, west_east_map, H_dict, course_H_dict)
    SGSTAP_Model.to(device)
    print(SGSTAP_Model)
    num_of_parameters = 0
    for name, parameters in SGSTAP_Model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)
    trainer = optim.Adam(SGSTAP_Model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)
    risk_mask = get_mask(mask_filename)
    best_rmse, best_recall, best_map, best_mae, best_mape, best_r_2 = training(
        SGSTAP_Model,
        training_epoch,
        train_loader,
        val_loader,
        test_loader,
        risk_mask,
        trainer,
        early_stop,
        device,
        scaler,
        data_type=config['data_type']
    )
    return best_rmse, best_recall, best_map, best_mae, best_mape, best_r_2


if __name__ == "__main__":

    main(config)
