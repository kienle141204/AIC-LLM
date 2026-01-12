import torch
import numpy as np
import torch.utils.data
from utils.utils import get_adjacency_matrix_2direction, get_adjacency_matrix
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from data.scaler import StandardScaler, MinMaxScaler
import os

def generate_sample_by_sliding_window(data, sample_len, step=1):
    sample = []
    for i in range(0, data.shape[0] - sample_len, step):
        sample.append(torch.unsqueeze(data[i:i+sample_len], 0))
    
    if (data.shape[0] - sample_len) % step != 0:
        sample.append(torch.unsqueeze(data[-sample_len:], 0))
    
    sample = torch.concat(sample, dim=0)
    return sample


class BasicDataset(torch.utils.data.Dataset):

    history: torch.Tensor        # (B, sample_len, node_num, features)
    history_avg: torch.Tensor    # (B, sample_len, node_num, features) - weekly average
    target: torch.Tensor         # (B, output_len, node_num, output_dim)
    target_avg: torch.Tensor     # (B, output_len, node_num, output_dim) - weekly average target
    timestamp: torch.Tensor      # (B, window_size, 5)

    def __init__(self, history, history_avg, target, target_avg, timestamp, training=False) -> None:
        self.history = history
        self.history_avg = history_avg
        self.target = target
        self.target_avg = target_avg
        self.timestamp = timestamp
        self.training = training

    def __len__(self):
        return self.history.shape[0]
    
    def __getitem__(self, index):
        return (self.history[index], self.history_avg[index], 
                self.target[index], self.target_avg[index], self.timestamp[index])


class DataProvider():
    node_num: int
    features: int
    data: torch.Tensor       # (T, node_num, features) - dữ liệu gốc
    timestamp: torch.Tensor  # (T, 5) - [month, day, weekday, hour, minute]

    def __init__(self, data_path, adj_path, dataset, node_shuffle_seed=None) -> None:
        self.dataset = dataset

        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp = self.read_data(data_path, adj_path)

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:, idx, :]
            self.adj_mx = self.adj_mx[idx, :][:, idx]

    def compute_weekly_average(self, train_data, train_timestamp):
        T_full = self.data.shape[0]
        
        # Tính weekdaytime cho tập train
        weekday = train_timestamp[:, 2]      # 0-6
        hour = train_timestamp[:, 3]         # 0-23
        minute = train_timestamp[:, 4]       # 0-59
        train_weekdaytime = weekday * 288 + (hour * 60 + minute) // 5

        weekly_avg_dict = {}
        
        # Tính trung bình cho từng node, từng feature
        for node_idx in range(self.node_num):
            for feature_idx in range(self.features):

                flow_values = train_data[:, node_idx, feature_idx].cpu().numpy()
                wdt_values = train_weekdaytime.cpu().numpy()

                df = pd.DataFrame({
                    'weekdaytime': wdt_values,
                    'flow': flow_values
                })
                
                def get_mean_without_null(data):
                    return data[data != 0].mean() if (data != 0).any() else 0
                
                avg_by_wdt = df.groupby('weekdaytime')['flow'].apply(get_mean_without_null)
                
                for wdt, avg_val in avg_by_wdt.items():
                    weekly_avg_dict[(node_idx, feature_idx, wdt)] = avg_val
        
        weekly_avg_full = torch.zeros_like(self.data)
        
        full_weekday = self.timestamp[:, 2]
        full_hour = self.timestamp[:, 3]
        full_minute = self.timestamp[:, 4]
        full_wdt = (full_weekday * 288 + (full_hour * 60 + full_minute) // 5).cpu().numpy()
        
        for t in range(T_full):
            wdt = full_wdt[t]
            for node_idx in range(self.node_num):
                for feature_idx in range(self.features):
                    key = (node_idx, feature_idx, wdt)
                    weekly_avg_full[t, node_idx, feature_idx] = weekly_avg_dict.get(key, 0)
        
        return weekly_avg_full

    def getdataset(self, sample_len, output_len, window_size,
                   input_dim, output_dim,
                   train_ratio, val_ratio, few_shot=1):
        
        self.data = self.data.float().cuda()
        self.timestamp = self.timestamp.cuda()

        all_len = self.data.shape[0]
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0, int(train_len * few_shot)]
        val_range = [train_len, train_len + val_len]
        test_range = [train_len + val_len, all_len]

        train_data = self.data[train_range[0]:train_range[1]]
        train_te = self.timestamp[train_range[0]:train_range[1]]
        
        weekly_avg = self.compute_weekly_average(train_data, train_te)
        weekly_avg = weekly_avg.cuda()

        scaler_data = self.data[train_range[0]:train_range[1]]
        dim = scaler_data.shape[-1]
        mean = [scaler_data[..., i:i+1].mean() for i in range(dim)]
        std = [scaler_data[..., i:i+1].std() for i in range(dim)]
        self.scaler = self.getscalerclass()(mean, std)

        # Training set
        train_data_normal = self.data[train_range[0]:train_range[1]]
        train_data_avg = weekly_avg[train_range[0]:train_range[1]]
        
        train_sample = generate_sample_by_sliding_window(train_data_normal, sample_len=window_size)
        train_sample_avg = generate_sample_by_sliding_window(train_data_avg, sample_len=window_size)
        
        train_x = train_sample[:, :sample_len, ..., :input_dim]
        train_x_avg = train_sample_avg[:, :sample_len, ..., :input_dim]
        train_y_avg = train_sample_avg[:, -output_len:, ..., :output_dim]
        train_y = train_sample[:, -output_len:, ..., :output_dim]
        
        train_x = self.scaler.transform(train_x)
        train_x_avg = self.scaler.transform(train_x_avg)
        
        train_te = generate_sample_by_sliding_window(train_te, sample_len=window_size)
        train_dataset = BasicDataset(history=train_x, history_avg=train_x_avg, 
                                    target=train_y, target_avg=train_y_avg, timestamp=train_te, training=True)

        # Validation set
        val_data_normal = self.data[val_range[0]:val_range[1]]
        val_data_avg = weekly_avg[val_range[0]:val_range[1]]
        val_te = self.timestamp[val_range[0]:val_range[1]]
        
        val_sample = generate_sample_by_sliding_window(val_data_normal, sample_len=window_size)
        val_sample_avg = generate_sample_by_sliding_window(val_data_avg, sample_len=window_size)
        
        val_x = val_sample[:, :sample_len, ..., :input_dim]
        val_x_avg = val_sample_avg[:, :sample_len, ..., :input_dim]
        val_y_avg = val_sample_avg[:, -output_len:, ..., :output_dim]
        val_y = val_sample[:, -output_len:, ..., :output_dim]
        
        val_x = self.scaler.transform(val_x)
        val_x_avg = self.scaler.transform(val_x_avg)
        
        val_te = generate_sample_by_sliding_window(val_te, sample_len=window_size)
        val_dataset = BasicDataset(history=val_x, history_avg=val_x_avg, 
                                   target=val_y, target_avg=val_y_avg, timestamp=val_te)

        # Test set
        test_data_normal = self.data[test_range[0]:test_range[1]]
        test_data_avg = weekly_avg[test_range[0]:test_range[1]]
        test_te = self.timestamp[test_range[0]:test_range[1]]
        
        test_sample = generate_sample_by_sliding_window(test_data_normal, sample_len=window_size)
        test_sample_avg = generate_sample_by_sliding_window(test_data_avg, sample_len=window_size)
        
        test_x = test_sample[:, :sample_len, ..., :input_dim]
        test_x_avg = test_sample_avg[:, :sample_len, ..., :input_dim]
        test_y_avg = test_sample_avg[:, -output_len:, ..., :output_dim]
        test_y = test_sample[:, -output_len:, ..., :output_dim]
        
        test_x = self.scaler.transform(test_x)
        test_x_avg = self.scaler.transform(test_x_avg)
        
        test_te = generate_sample_by_sliding_window(test_te, sample_len=window_size)
        test_dataset = BasicDataset(history=test_x, history_avg=test_x_avg, 
                                    target=test_y, target_avg=test_y_avg, timestamp=test_te)

        return train_dataset, val_dataset, test_dataset

    def getadj(self):
        return self.adj_mx, self.distance_mx
    
    def getscalerclass(self):
        return StandardScaler


def generatetimestamp(start, periods, freq):
    time = pd.date_range(start=start, periods=periods, freq=freq)
    
    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))
    
    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)
    timestamp = torch.tensor(timestamp)
    
    return timestamp


timestampfun = {
    'PEMS08': lambda T: generatetimestamp(start='20160701 00:00:00', periods=T, freq='5min'),
    'PEMS07': lambda T: generatetimestamp(start='20170501 00:00:00', periods=T, freq='5min'),
    'PEMS04': lambda T: generatetimestamp(start='20180101 00:00:00', periods=T, freq='5min'),
    'PEMS03': lambda T: generatetimestamp(start='20180901 00:00:00', periods=T, freq='5min'),
    'NYCTAXI': lambda T: generatetimestamp(start='20160401 00:00:00', periods=T, freq='30min'),
    'CHIBIKE': lambda T: generatetimestamp(start='20160401 00:00:00', periods=T, freq='30min'),
}


class PEMSFLOWProvider(DataProvider):

    def read_data(self, data_path, adj_path=None) -> None:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])
        
        T, node_num, features = data.shape
        
        if 'PEMS03' in self.dataset:
            id_filename = adj_path.replace('csv', 'txt')
        else:
            id_filename = None
        
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num, id_filename)
        adj_mx = np.where(np.eye(node_num).astype('bool'), 1, adj_mx)
        
        timestamp = timestampfun[self.dataset[:6]](T)
        
        return data, node_num, features, adj_mx, distance_mx, timestamp


class NYCTAXIProvider(DataProvider):

    def read_data(self, data_path, adj_path=None) -> None:
        data = torch.from_numpy(np.load(data_path)['data'][..., :])
        data = np.transpose(data, (1, 0, 2))
        
        T, node_num, features = data.shape
        
        adj_mx = np.ones((node_num, node_num)).astype(np.float32)
        distance_mx = np.ones((node_num, node_num)).astype(np.float32)
        timestamp = timestampfun[self.dataset](T)
        
        return data, node_num, features, adj_mx, distance_mx, timestamp