import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn.mixture import BayesianGaussianMixture
import skfuzzy as fuzz

# ========== 共享最近邻（SNN）相似度计算 ==========
# 全局缓存，避免重复计算SNN矩阵
_SNN_CACHE = {}

def calculate_snn_similarity(X, k=15, use_cache=True):
    """
    计算样本间的共享最近邻相似度，增加缓存机制避免重复计算
    :param X: 特征矩阵 (n_samples, n_features)
    :param k: 最近邻数量
    :param use_cache: 是否使用缓存
    :return: SNN相似度矩阵 (n_samples, n_samples)
    """
    cache_key = (hash(X.data.tobytes()), k) if use_cache else None
    
    if use_cache and cache_key in _SNN_CACHE:
        return _SNN_CACHE[cache_key]
    
    n_samples = X.shape[0]
    # 不限制k值，保持最高精度计算
    # 移除大样本量降k逻辑，完全追求最优效果
    k = 15 if n_samples > 20 else min(10, n_samples-1)
    
    # 1. 找到每个样本的k个最近邻，启用多进程加速
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    # 2. 构建SNN相似度矩阵 - 优化计算逻辑
    snn_sim = np.zeros((n_samples, n_samples), dtype=np.float16)  # 使用半精度浮点数减少内存占用
    for i in range(n_samples):
        neighbors_i = indices[i]
        for j in neighbors_i:
            if j > i:
                shared = len(np.intersect1d(neighbors_i, indices[j], assume_unique=True))
                sim = shared / k
                snn_sim[i, j] = sim
                snn_sim[j, i] = sim
    
    if use_cache:
        _SNN_CACHE[cache_key] = snn_sim
    
    return snn_sim

def clear_snn_cache():
    """清空SNN缓存，释放内存"""
    global _SNN_CACHE
    _SNN_CACHE = {}

def snn_clustering_score(X, labels, k=10, snn_sim=None):
    """
    基于SNN相似度的聚类质量评估分数（越高越好）
    计算每个簇内的平均SNN相似度和簇间的平均SNN相似度的比值
    :param X: 特征矩阵
    :param labels: 聚类标签
    :param k: 最近邻数量
    :param snn_sim: 预计算的SNN相似度矩阵，不传则自动计算
    :return: SNN聚类质量分数
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2 or n_clusters >= len(labels):
        return -1.0
    
    # 优先使用传入的预计算SNN矩阵
    if snn_sim is None:
        snn_sim = calculate_snn_similarity(X, k=k)
    
    # 计算簇内平均相似度
    intra_sim = 0.0
    valid_clusters = 0
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_size = np.sum(cluster_mask)
        if cluster_size < 2:
            continue
        # 取簇内所有样本的SNN相似度的平均值
        cluster_sim = snn_sim[cluster_mask][:, cluster_mask]
        # 排除对角线元素（自己和自己的相似度）
        np.fill_diagonal(cluster_sim, 0)
        intra_sim += np.sum(cluster_sim) / (cluster_size * (cluster_size - 1))
        valid_clusters += 1
    
    if valid_clusters == 0:
        return -1.0
    
    intra_sim /= valid_clusters
    
    # 计算簇间平均相似度
    inter_sim = 0.0
    count = 0
    for i in range(valid_clusters):
        mask_i = (labels == unique_labels[i])
        for j in range(i + 1, valid_clusters):
            mask_j = (labels == unique_labels[j])
            inter_sim += np.mean(snn_sim[mask_i][:, mask_j])
            count += 1
    
    inter_sim /= max(count, 1e-6)
    
    # 最终得分 = 簇内相似度 / (簇间相似度 + 1e-6)
    return intra_sim / (inter_sim + 1e-6)

def combined_clustering_score(X, labels, silhouette_weight=0.3, snn_weight=0.7, k=15, snn_sim=None):
    """
    综合聚类评估分数：结合轮廓系数和SNN分数
    :param X: 特征矩阵
    :param labels: 聚类标签
    :param silhouette_weight: 轮廓系数权重
    :param snn_weight: SNN分数权重
    :param k: SNN最近邻数量
    :param snn_sim: 预计算的SNN相似度矩阵，不传则自动计算
    :return: 综合分数，越高越好
    """
    sil_score = silhouette_score(X, labels)
    snn_score = snn_clustering_score(X, labels, k=k, snn_sim=snn_sim)
    
    # 归一化SNN分数到[0,1]区间
    norm_snn = min(snn_score / 10, 1.0) if snn_score > 0 else 0
    
    return sil_score * silhouette_weight + norm_snn * snn_weight

# ========== 统一自动调参入口 ==========
def auto_select_params(X, algorithm_name, n_true_clusters=None):
    """统一自动调参入口（已删除基于真实样本的逻辑，改用SNN相似度评估）"""
    # 先清空缓存
    clear_snn_cache()
    
    if algorithm_name == 'kmeans':
        params = _select_kmeans(X)
    elif algorithm_name == 'bgmm':
        params = _select_bgmm(X)
    elif algorithm_name == 'fcm':
        params = _select_fcm(X)
    elif algorithm_name == 'meanshift':
        params = _select_meanshift(X)
    else:
        params = {}
    
    # 最后清空缓存释放内存
    clear_snn_cache()
    return params

# ========== 各算法内部调参逻辑 ==========
def _select_kmeans(X):
    best_k, best_score = 2, -1
    max_k = min(20, int(len(X) / 5))  # 扩大最大聚类数搜索范围，最高20类
    # 预计算SNN矩阵，只算一次
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_k + 1):
        # 增加n_init到20次，避免局部最优
        labels = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=20, max_iter=1000).fit_predict(X)
        score = combined_clustering_score(X, labels, snn_sim=snn_sim)
        if score > best_score: 
            best_score, best_k = score, k
    return {'n_clusters': best_k}

def _select_bgmm(X):
    best_k, best_score = 2, -1
    max_k = min(20, int(len(X) / 5))  # 扩大最大聚类数搜索范围
    # 预计算SNN矩阵，只算一次
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_k + 1):
        m = BayesianGaussianMixture(
            n_components=k, 
            random_state=42,
            n_init=10,  # 增加初始化次数到10次，避免局部最优
            max_iter=1000,  # 增加迭代次数到1000次，保证完全收敛
            tol=1e-6  # 降低收敛阈值，提高精度
        ).fit(X)
        score = combined_clustering_score(X, m.predict(X), snn_sim=snn_sim)
        if score > best_score: 
            best_score, best_k = score, k
    return {'n_clusters': best_k}

def _select_fcm(X):
    best_k, best_m, best_score = 2, 2.0, -1
    max_k = min(20, int(len(X) / 5))  # 扩大最大聚类数搜索范围
    X_T = X.T
    # 预计算SNN矩阵，只算一次
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_k + 1):
        # 扩大m的搜索范围，步长更小，覆盖更多可能
        for m in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X_T, c=k, m=m, error=1e-7, maxiter=2000, seed=42  # 大幅降低error阈值，增加迭代次数，完全收敛
            )
            labels = np.argmax(u, axis=0)
            if len(np.unique(labels)) < 2: 
                continue
            score = combined_clustering_score(X, labels, snn_sim=snn_sim)
            if score > best_score: 
                best_score, best_k, best_m = score, k, m
    return {'n_clusters': best_k, 'm': best_m}

def _select_meanshift(X):
    best_bw = None
    best_score = -1
    best_n_clusters = 2
    
    # 预计算SNN矩阵，只算一次
    snn_sim = calculate_snn_similarity(X)
    
    # 大幅增加quantile搜索点，步长更小，覆盖更多可能
    quantiles = [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.025, 0.028, 
                0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    for q in quantiles:
        # 移除采样限制，使用全部样本计算带宽
        bw = estimate_bandwidth(X, quantile=q, n_samples=None, random_state=42)
        # 禁用bin_seeding，使用更精确的聚类
        labels = MeanShift(bandwidth=bw, bin_seeding=False, n_jobs=-1, max_iter=1000).fit_predict(X)
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2 or n_clusters > min(25, int(len(X)/5)):  # 扩大最大聚类数限制
            continue
            
        score = combined_clustering_score(X, labels, snn_sim=snn_sim)
        
        # 优先选择聚类数在合理范围内且评分最高的
        if score > best_score:
            best_score = score
            best_bw = bw
            best_n_clusters = n_clusters
    
    return {'bandwidth': best_bw}

# ========== 单独导出的兼容函数（解决未定义报错） ==========
def auto_select_kmeans_params(X, max_k=10):
    """KMeans单独调参函数，兼容原有调用"""
    clear_snn_cache()
    best_k, best_score = 2, -1
    max_k = min(max_k, int(len(X)/10))
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_k+1):
        labels = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto').fit_predict(X)
        score = combined_clustering_score(X, labels, snn_sim=snn_sim)
        if score > best_score: 
            best_score, best_k = score, k
    clear_snn_cache()
    return best_k, best_score

def auto_select_bgmm_params(X, max_components=10):
    """BGMM单独调参函数，解决本次未定义报错"""
    clear_snn_cache()
    best_k, best_score = 2, -1
    max_components = min(max_components, int(len(X)/10))
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_components+1):
        m = BayesianGaussianMixture(
            n_components=k, 
            random_state=42,
            n_init=3,
            max_iter=300
        ).fit(X)
        score = combined_clustering_score(X, m.predict(X), snn_sim=snn_sim)
        if score > best_score: 
            best_score, best_k = score, k
    clear_snn_cache()
    return best_k, best_score

def auto_select_fcm_params(X, max_k=10):
    """FCM单独调参函数，兼容原有调用"""
    clear_snn_cache()
    best_k, best_m, best_score = 2, 2.0, -1
    max_k = min(max_k, int(len(X)/10))
    X_T = X.T
    snn_sim = calculate_snn_similarity(X)
    for k in range(2, max_k+1):
        for m in [1.5, 2.0, 2.5]:
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X_T, c=k, m=m, error=1e-5, maxiter=400, seed=42
            )
            labels = np.argmax(u, axis=0)
            if len(np.unique(labels)) < 2: 
                continue
            score = combined_clustering_score(X, labels, snn_sim=snn_sim)
            if score > best_score: 
                best_score, best_k, best_m = score, k, m
    clear_snn_cache()
    return best_k, best_m, best_score

def auto_select_meanshift_params(X, n_true_clusters=None):
    """MeanShift单独调参函数，兼容原有调用"""
    clear_snn_cache()
    best_bw = None
    best_quantile = 0.2
    best_score = -1
    snn_sim = calculate_snn_similarity(X)
    # 回调quantile搜索点
    for q in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        bw = estimate_bandwidth(X, quantile=q, n_samples=min(400, len(X)), random_state=42)
        model = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
        labels = model.fit_predict(X)
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters > min(15, int(len(X)/8)):
            continue
        score = combined_clustering_score(X, labels, snn_sim=snn_sim)
        if score > best_score:
            best_score = score
            best_bw = bw
            best_quantile = q
    clear_snn_cache()
    return best_bw, best_quantile, best_score