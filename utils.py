import os
import torch
import random
import argparse
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()
    print(string)


def count_parameters(model):
    """统计模型参数"""
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""图相关"""


# def get_adjacency_matrix(num_of_vertices, type_='connectivity'):



# def construct_adj(A, steps):
#     """
#     构建local 时空图
#     :param A: np.ndarray, adjacency matrix, shape is (N, N)
#     :param steps: 选择几个时间步来构建图
#     :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
#     """
#     N = len(A)  # 获得行数
#     adj = np.zeros((N * steps, N * steps))
#
#     for i in range(steps):
#         """对角线代表各个时间步自己的空间图，也就是A"""
#         adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
#
#     for i in range(N):
#         for k in range(steps - 1):
#             """每个节点只会连接相邻时间步的自己"""
#             adj[k * N + i, (k + 1) * N + i] = 1
#             adj[(k + 1) * N + i, k * N + i] = 1
#
#     for i in range(len(adj)):
#         """加入自回"""
#         adj[i, i] = 1
#
#     return adj


"""数据加载器"""


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        数据加载器
        :param xs:训练数据
        :param ys:标签数据
        :param batch_size:batch大小
        :param pad_with_last_sample:剩余数据不够时，是否复制最后的sample以达到batch大小
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        """洗牌"""
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """标准转换器"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


def load_dataset(dataset_dir, normalizer, batch_size, valid_batch_size=None, test_batch_size=None):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param valid_batch_size: 验证集batch大小
    :param test_batch_size: 测试集batch大小
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    if normalizer == 'max01':
        minimum = data['x_train'].min()
        maximum = data['x_train'].max()

        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')

    elif normalizer == 'max11':
        minimum = data['x_train'].min()
        maximum = data['x_train'].max()

        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')

    elif normalizer == 'std':
        mean = data['x_train'].mean()
        std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def load_dataset_noVal(dataset_dir, normalizer, batch_size, test_batch_size=None):
    """
    加载数据集
    :param dataset_dir: 数据集目录
    :param normalizer: 归一方式
    :param batch_size: batch大小
    :param test_batch_size: 测试集batch大小
    """
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    data['x_test'] = np.concatenate((data['x_val'], data['x_test']))
    data['y_test'] = np.concatenate((data['y_val'], data['y_test']))

    if normalizer == 'std':
        mean = data['x_train'].mean()
        std = data['x_train'].std()

        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')

    elif normalizer == 'None':
        scaler = NScaler()
        print('Does not normalize the dataset')
    else:
        raise ValueError

    for category in ['train', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


"""指标"""


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)

    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def quadrant_classification(value):
    if value[0] > 5 and value[1] > 5:
        return 0  # Quadrant 1
    elif value[0] <= 5 and value[1] > 5:
        return 1  # Quadrant 2
    elif value[0] <= 5 and value[1] <= 5:
        return 2  # Quadrant 3
    else:
        return 3  # Quadrant 4


def bi_classification(value):
    if value > 5:
        return 1
    else:
        return 0


def calculate_accuracy(predictions, labels):
    assert predictions.shape == labels.shape, "Predictions and labels must have the same shape"
    assert predictions.shape[1] == 2, "Each item must have two columns (valence and arousal)"

    valence_correct = 0
    arousal_correct = 0
    both_correct = 0
    accu = {}

    for pred, label in zip(predictions, labels):
        valence_diff = abs(pred[0] - label[0])
        arousal_diff = abs(pred[1] - label[1])

        if valence_diff <= 0.5:
            valence_correct += 1
        if arousal_diff <= 0.5:
            arousal_correct += 1
        if valence_diff <= 0.5 and arousal_diff <= 0.5:
            both_correct += 1

    total_samples = len(predictions)
    accu['valence'] = valence_correct / total_samples
    accu['arousal'] = arousal_correct / total_samples
    accu['all'] = both_correct / total_samples

    return accu


def accuracy(preds, labels):
    accu = {}
    preds_class = torch.tensor([quadrant_classification(value) for value in preds])
    labels_class = torch.tensor([quadrant_classification(value) for value in labels])

    valence_preds = torch.tensor([bi_classification(value) for value in preds[:, 0]])
    valence_labels = torch.tensor([bi_classification(value) for value in labels[:, 0]])
    arousal_preds = torch.tensor([bi_classification(value) for value in preds[:, 1]])
    arousal_labels = torch.tensor([bi_classification(value) for value in labels[:, 1]])

    correct_all = torch.sum(preds_class == labels_class).float()
    correct_val = torch.sum(valence_preds == valence_labels).float()
    correct_arou = torch.sum(arousal_preds == arousal_labels).float()
    total = len(labels)  # Since each element is a pair of values

    accu['all'] = correct_all / total
    accu['valence'] = correct_val / total
    accu['arousal'] = correct_arou / total
    return accu


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    acc_all = accuracy(pred, real)['all']
    acc_val = accuracy(pred, real)['valence']
    acc_arou = accuracy(pred, real)['arousal']
    return mae, mape, rmse, acc_all, acc_val, acc_arou







