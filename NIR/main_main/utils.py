import pywt
import numpy as np
import pandas as pd
import networkx as nx
from scipy.fft import fft, rfft
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew

# Пример функций для создания признаков
def mean_feature(data):
    """Среднее значение."""
    return np.mean(data, axis=0)

def median_feature(data):
    """Медиана."""
    return np.median(data, axis=0)

def std_feature(data):
    """Стандартное отклонение."""
    return np.std(data, axis=0)

def min_feature(data):
    """Минимум."""
    return np.min(data, axis=0)

def max_feature(data):
    """Максимум."""
    return np.max(data, axis=0)

def kurtosis_feature(data):
    """Эксцесс."""
    return kurtosis(data, axis=0)

def skew_feature(data):
    """Асимметрия."""
    return skew(data, axis=0)

def quantile_feature(data, q=0.25):
    """Квантиль."""
    return np.quantile(data, q, axis=0)

def weekday_feature(data, index):
    """День недели."""
    return np.array([index.weekday] * data.shape[1]).T

def hour_feature(data, index):
    """Час дня."""
    return np.array([index.hour] * data.shape[1]).T

def peak_hours_feature(data, index):
    """Пиковые часы."""
    nodes = data.shape[1]  # Количество узлов
    peak_hours = np.isin(index.hour, [7, 8, 9, 17, 18, 19]).astype(int)  # Векторизованная проверка
    return np.tile(peak_hours[:, np.newaxis], (1, nodes))  # Растягиваем на все узлы

def is_holiday_feature(data, index, holidays):
    """Выходные/праздники."""
    return np.array([1 if index.date() in holidays else 0] * data.shape[1]).T

def season_feature(data, index):
    """Сезонность (время года)."""
    month = index.month
    season = (month % 12) // 3
    return np.array([season] * data.shape[1]).T

def rolling_mean_feature(data, window=12):
    """Скользящее среднее."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).mean().values

def rolling_std_feature(data, window=12):
    """Скользящее стандартное отклонение."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).std().fillna(0).values

def rolling_min_feature(data, window=12):
    """Скользящий минимум."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).min().values

def rolling_max_feature(data, window=12):
    """Скользящий максимум."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).max().values

def fft_feature(data):
    """Преобразование Фурье (FFT)."""
    return np.abs(rfft(data, axis=0)) # type: ignore

def dwt_feature(data, wavelet='db1'):
    """Вейвлет-преобразование (DWT)."""
    coeffs = pywt.wavedec(data, wavelet, axis=0)
    result = np.concatenate(coeffs, axis=0)
    return result[:data.shape[0]]

def diff_feature(data, periods=1):
    """Разность значений."""
    return pd.DataFrame(data).diff(periods=periods).fillna(0).values

def autocorr_feature(data, lag=1):
    """Автокорреляция для каждого столбца данных."""
    df = pd.DataFrame(data)
    autocorr_values = df.apply(lambda col: col.autocorr(lag=lag)).fillna(0).values
    return autocorr_values



def degree_feature(adj_matrix):
    """Степень узла."""
    G = nx.from_numpy_array(adj_matrix)
    return dict(G.degree())

def weighted_degree_feature(adj_matrix):
    """Взвешенная степень узла."""
    G = nx.from_numpy_array(adj_matrix)
    return dict(G.degree(weight='weight'))

def degree_centrality_feature(adj_matrix):
    """Степень центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.degree_centrality(G)

def closeness_centrality_feature(adj_matrix):
    """Близость центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.closeness_centrality(G)

def betweenness_centrality_feature(adj_matrix):
    """Посредничество центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.betweenness_centrality(G)

def eigenvector_centrality_feature(adj_matrix):
    """Собственный вектор центральности."""
    G = nx.from_numpy_array(adj_matrix)
    try:
        return nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality computation did not converge.")
        return {node: 0 for node in G.nodes()}

def edge_weight_feature(adj_matrix):
    """Вес узлов, основанный на сумме весов инцидентных ребер."""
    G = nx.from_numpy_array(adj_matrix)
    node_weights = {node: 0 for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)  # Использовать вес из данных или 1 по умолчанию
        node_weights[u] += weight
        node_weights[v] += weight
    return node_weights  # Возвращаем веса в виде словаря {node: weight}

def shortest_path_length_feature(adj_matrix):
    """Расстояние между узлами."""
    G = nx.from_numpy_array(adj_matrix)
    return {source: dict(lengths) for source, lengths in nx.shortest_path_length(G)}

def graph_density_feature(adj_matrix):
    """Плотность графа для каждого узла (глобальная плотность одинакова для всех узлов)."""
    G = nx.from_numpy_array(adj_matrix)
    density = nx.density(G)
    nodes = G.nodes()
    return {node: density for node in nodes}  # Одинаковая плотность для всех узлов


def graph_diameter_feature(adj_matrix):
    """Диаметр графа. Возвращает значение для всех узлов, если граф связный."""
    G = nx.from_numpy_array(adj_matrix)
    if not nx.is_connected(G):
        print("Graph is not connected; diameter is infinite.")
        diameter = float('inf')
    else:
        diameter = nx.diameter(G)
    nodes = G.nodes()
    return {node: diameter for node in nodes}  # Одинаковый диаметр для всех узлов


def clustering_coefficient_feature(adj_matrix):
    """Коэффициент кластеризации для каждого узла."""
    G = nx.from_numpy_array(adj_matrix)
    clustering = nx.clustering(G)  # Словарь с коэффициентами кластеризации для каждого узла
    return clustering