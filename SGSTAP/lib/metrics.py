import numpy as np
import math


def transfer_dtype(y_true, y_pred):
    return y_true.astype('float32'), y_pred.astype('float32')


def mask_mse_np(y_true, y_pred, region_mask, null_val=None):
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0, 1, 0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    return np.mean(((y_true - y_pred) * mask) ** 2)


def mask_rmse_np(y_true, y_pred, region_mask, null_val=None):
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    return math.sqrt(mask_mse_np(y_true, y_pred, region_mask, null_val))


def nonzero_num(y_true):
    nonzero_list = []
    threshold = 0
    for i in range(len(y_true)):
        non_zero_nums = (y_true[i] > threshold).sum()
        nonzero_list.append(non_zero_nums)
    return nonzero_list


def get_top(data, accident_nums):
    data = data.reshape((data.shape[0], -1))
    topk_list = []
    for i in range(len(data)):
        risk = {}
        for j in range(len(data[i])):
            risk[j] = data[i][j]
        k = int(accident_nums[i])
        topk_list.append(list(dict(sorted(risk.items(), key=lambda x: x[1], reverse=True)[:k]).keys()))
    return topk_list


def Recall(y_true, y_pred, region_mask):
    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask

    accident_grids_nums = nonzero_num(tmp_y_true)

    true_top_k = get_top(tmp_y_true, accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred, accident_grids_nums)

    hit_sum = 0
    for i in range(len(true_top_k)):
        intersection = [v for v in true_top_k[i] if v in pred_top_k[i]]
        hit_sum += len(intersection)
    return hit_sum / sum(accident_grids_nums) * 100


def MAP(y_true, y_pred, region_mask):
    region_mask = np.where(region_mask >= 1, 0, -1000)
    tmp_y_true = y_true + region_mask
    tmp_y_pred = y_pred + region_mask
    accident_grids_nums = nonzero_num(tmp_y_true)
    true_top_k = get_top(tmp_y_true, accident_grids_nums)
    pred_top_k = get_top(tmp_y_pred, accident_grids_nums)
    all_k_AP = []
    for sample in range(len(true_top_k)):
        all_k_AP.append(AP(list(true_top_k[sample]), list(pred_top_k[sample])))
    return sum(all_k_AP) / len(all_k_AP)


def AP(label_list, pre_list):
    hits = 0
    sum_precs = 0
    for n in range(len(pre_list)):
        if pre_list[n] in label_list:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(label_list)
    else:
        return 0


def MAE(y_true, y_pred, region_mask, null_val):
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0, 1, 0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    absolute_error = np.abs((y_true - y_pred) * mask)
    return np.mean(absolute_error)


def MAPE(y_true, y_pred, region_mask, null_val):
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0, 1, 0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    epsilon = 1e-8
    percentage_error = np.abs((y_true - y_pred) * mask / (y_true + epsilon)) * 100
    return np.mean(percentage_error)


def R_2(y_true, y_pred, region_mask, null_val):
    y_true, y_pred = transfer_dtype(y_true, y_pred)
    if null_val is not None:
        label_mask = np.where(y_true > 0, 1, 0).astype('float32')
        mask = region_mask * label_mask
    else:
        mask = region_mask
    mask /= mask.mean()
    y_true = y_true * mask
    y_pred = y_pred * mask
    total_error = y_true - np.mean(y_true)
    total_squared_error = np.sum(total_error ** 2)
    residual_error = y_true - y_pred
    residual_squared_error = np.sum(residual_error ** 2)
    return 1 - (residual_squared_error / total_squared_error)


def mask_evaluation_np(y_true, y_pred, region_mask, null_val=None):
    rmse_ = mask_rmse_np(y_true, y_pred, region_mask, null_val)
    recall_ = Recall(y_true, y_pred, region_mask)
    map_ = MAP(y_true, y_pred, region_mask)
    mae = MAE(y_true, y_pred, region_mask, null_val)
    mape = MAPE(y_true, y_pred, region_mask, null_val)
    r_2 = R_2(y_true, y_pred, region_mask, null_val)
    return rmse_, recall_, map_, mae, mape, r_2