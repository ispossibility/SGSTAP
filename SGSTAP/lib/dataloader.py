import numpy as np
import pickle as pkl
import configparser
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from lib.utils import Scaler_NYC, Scaler_Chi

def split_and_norm_data_time(all_data,
                             course_data,
                             train_rate=0.6,
                             valid_rate=0.2,
                             recent_prior=3,
                             week_prior=4,
                             one_day_period=24,
                             days_of_week=7,
                             pre_len=1):
    num_of_time, channel, _, _ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate + valid_rate))
    for index, (start, end) in enumerate(((0, train_line), (train_line, valid_line), (valid_line, num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
                course_scaler = Scaler_NYC(course_data[start:end, :, :, :])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
                course_scaler = Scaler_NYC(course_data[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        course_norm_data = course_scaler.transform(course_data[start:end, :, :, :])
        X, Y = [], []
        Course_X, Course_Y = [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = norm_data[period_list, :, :, :]
            X.append(feature)
            Y.append(label)
        for i in range(len(course_norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = course_norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = course_norm_data[period_list, :, :, :]
            Course_X.append(feature)
            Course_Y.append(label)
        yield np.array(X), np.array(Y), np.array(Course_X), np.array(Course_Y), scaler


def normal_and_generate_dataset_time(
        all_data_filename,
        course_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)
    course_data = pkl.load(open(course_data_filename, 'rb')).astype(np.float32)
    for i in split_and_norm_data_time(all_data,
                                      course_data,
                                      train_rate=train_rate,
                                      valid_rate=valid_rate,
                                      recent_prior=recent_prior,
                                      week_prior=week_prior,
                                      one_day_period=one_day_period,
                                      days_of_week=days_of_week,
                                      pre_len=pre_len):
        yield i


def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename

    Returns:
        {np.array} -- mask matrix，维度(W,H)
    """
    mask = pkl.load(open(mask_path, 'rb')).astype(np.float32)
    return mask


def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path

    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path, 'rb')).astype(np.float32)
    return adjacent


def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename

    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path, 'rb')).astype(np.float32)
    return grid_node_map 
