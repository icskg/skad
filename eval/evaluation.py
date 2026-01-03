import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, \
    average_precision_score


def calculate_reconstruction_errors(x, out):
    """
    计算重构误差

    Args:
        x: 原始输入 [batch_size, seq_len, num_features]
        out: 模型输出 [batch_size, seq_len, num_features]

    Returns:
        torch.Tensor: 每个时间点的重构误差 [batch_size, seq_len]
    """
    # 使用MSE作为重构误差
    if isinstance(out, tuple) or isinstance(out, list):
        # 如果输出是元组或列表，取第一个元素作为重构结果
        reconstruction = out[0]
    else:
        reconstruction = out

    # 计算每个时间点每个特征的MSE，然后对特征维度求平均
    errors = torch.mean((x - reconstruction) ** 2, dim=-1)  # [batch_size, seq_len]
    return errors


def calculate_anomaly_metrics(errors, labels, adjustment_strategy='point_adjust'):
    """
    计算多变量时间序列异常检测性能指标（使用点调整策略）

    Args:
        errors: 重构误差 [num_samples, seq_len] 或 [num_samples * seq_len]
        labels: 真实标签 [num_samples, seq_len] 或 [num_samples * seq_len]
        adjustment_strategy: 调整策略，默认使用点调整

    Returns:
        dict: 包含各种性能指标的字典
    """
    # 确保输入是1D数组
    if errors.ndim > 1:
        errors = errors.reshape(-1)
    if labels.ndim > 1:
        labels = labels.reshape(-1)

    # 计算AUC-ROC和AUC-PR（不需要阈值）
    auc_roc = roc_auc_score(labels, errors) if labels.sum() > 0 else 0.5
    auc_pr = average_precision_score(labels, errors)

    # 确定阈值（使用误差的百分位数）
    threshold = np.percentile(errors, 95)  # 使用95%分位数作为阈值

    # 生成预测标签
    predictions = (errors > threshold).astype(int)

    # 应用点调整策略
    if adjustment_strategy == 'point_adjust':
        predictions = apply_point_adjustment(predictions, labels)

    far, _ = calculate_false_alarm_rate(labels, predictions, zero_division=0.0)
    # 计算性能指标
    metrics = {
        'f1_score': f1_score(labels, predictions, zero_division=0),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'accuracy': accuracy_score(labels, predictions),
        'threshold': threshold,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'far': far
    }

    return metrics


def apply_point_adjustment(predictions, labels):
    """
    应用点调整策略

    Args:
        predictions: 原始预测 [n]
        labels: 真实标签 [n]

    Returns:
        np.array: 调整后的预测
    """
    adjusted_predictions = predictions.copy()

    # 找到所有真实的异常段
    anomaly_segments = find_anomaly_segments(labels)

    for start, end in anomaly_segments:
        # 如果在这个异常段内至少有一个点被检测为异常
        segment_predictions = predictions[start:end + 1]
        if np.any(segment_predictions == 1):
            # 将这个异常段内的所有点都标记为异常
            adjusted_predictions[start:end + 1] = 1

    return adjusted_predictions


def find_anomaly_segments(labels):
    """
    找到标签中连续的异常段

    Args:
        labels: 真实标签数组

    Returns:
        list: 包含异常段(start, end)的列表
    """
    segments = []
    n = len(labels)
    i = 0

    while i < n:
        if labels[i] == 1:
            start = i
            # 找到连续的异常段
            while i < n and labels[i] == 1:
                i += 1
            end = i - 1
            segments.append((start, end))
        else:
            i += 1

    return segments

def calculate_false_alarm_rate(labels, predictions, zero_division: float = 0.0):

    # 计算混淆量
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))

    # 计算 FAR = FP / (FP + TN)（若没有负样本，设置 FAR = 0.0）
    neg_total = fp + tn
    if neg_total > 0:
        far = float(fp) / float(neg_total)
    else:
        far = zero_division

    return  far, (tp, fp, fn, tn)