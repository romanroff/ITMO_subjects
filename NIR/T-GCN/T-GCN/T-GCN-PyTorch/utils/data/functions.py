import numpy as np
import pandas as pd
import torch
from scipy.fft import fft, ifft, fftfreq

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj



def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, fourier=True, threshold=0.8
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :param fourier: whether to apply Fourier transform and denoise the data
    :param threshold: threshold for Fourier amplitude filtering (proportion of maximum amplitude)
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]

    if normalize:
        max_val = np.max(data)
        data = data / max_val

    if fourier:
        data = apply_fourier_denoising(data, threshold)
        print(f"Преобразование Фурье применено")

    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]

    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def apply_fourier_denoising(data, threshold):
    """
    Applies Fourier transform to the data, removes noise using a threshold, and returns the filtered data.
    
    :param data: 1D numpy array or 2D array (e.g., [time_steps, nodes])
    :param threshold: proportion of maximum amplitude to use as a cutoff for filtering
    :return: filtered data
    """
    if len(data.shape) == 1:
        # Single series (1D array)
        return denoise_series(data, threshold)
    else:
        # Multiple series (2D array, e.g., [time_steps, nodes])
        filtered_data = []
        for i in range(data.shape[1]):
            filtered_data.append(denoise_series(data[:, i], threshold))
        return np.array(filtered_data).T


def denoise_series(series, threshold):
    """
    Denoises a single time series using Fourier transform.
    
    :param series: 1D numpy array
    :param threshold: proportion of maximum amplitude to use as a cutoff for filtering
    :return: filtered series
    """
    fft_values = fft(series)
    amplitudes = np.abs(fft_values)
    max_amplitude = np.max(amplitudes)

    # Filter: keep only components above the threshold
    filtered_fft = np.zeros_like(fft_values)
    filtered_fft[amplitudes > (threshold * max_amplitude)] = fft_values[amplitudes > (threshold * max_amplitude)]

    # Inverse FFT to reconstruct the signal
    filtered_series = ifft(filtered_fft).real
    return filtered_series


# def generate_dataset(
#     data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
# ):
#     """
#     :param data: feature matrix
#     :param seq_len: length of the train data sequence
#     :param pre_len: length of the prediction data sequence
#     :param time_len: length of the time series in total
#     :param split_ratio: proportion of the training set
#     :param normalize: scale the data to (0, 1], divide by the maximum value in the data
#     :return: train set (X, Y) and test set (X, Y)
#     """
#     if time_len is None:
#         time_len = data.shape[0]
#     if normalize:
#         max_val = np.max(data)
#         data = data / max_val
#     train_size = int(time_len * split_ratio)
#     train_data = data[:train_size]
#     test_data = data[train_size:time_len]
#     train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
#     for i in range(len(train_data) - seq_len - pre_len):
#         train_X.append(np.array(train_data[i : i + seq_len]))
#         train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
#     for i in range(len(test_data) - seq_len - pre_len):
#         test_X.append(np.array(test_data[i : i + seq_len]))
#         test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
#     return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, fourier=False
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
        fourier=fourier
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
