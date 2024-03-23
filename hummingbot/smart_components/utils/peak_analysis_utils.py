from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks


def calculate_prominence(candles, prominence_percentage):
    price_range = candles['high'].max() - candles['low'].min()
    return price_range * prominence_percentage


def find_price_peaks(candles, prominence_nominal, distance):
    high_peaks, _ = find_peaks(candles['high'], prominence=prominence_nominal, distance=distance)
    low_peaks, _ = find_peaks(-candles['low'], prominence=prominence_nominal, distance=distance)
    return high_peaks, low_peaks


def hierarchical_clustering(peaks, num_clusters=3):
    Z = linkage(peaks.reshape(-1, 1), method='ward')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    centroids = [peaks[labels == k].mean() for k in range(1, num_clusters + 1)]
    return centroids, labels
