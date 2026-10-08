import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(X, scaler_path, pca_path=None, is_train=True, n_components=0.95):
    """
    数据预处理：标准化 + 可选PCA降维
    :param X: 原始特征数据
    :param scaler_path: scaler保存/加载路径
    :param pca_path: pca模型保存/加载路径，为None则不启用PCA降维
    :param is_train: 是否为训练模式（训练模式创建新scaler/pca，预测模式加载已有模型）
    :param n_components: PCA保留的方差比例，默认0.95保留95%以上的方差信息
    :return: 预处理后的特征数据
    """
    # 第一步：标准化处理（仅做数值缩放，不解决多重共线性问题）
    if is_train:
        # 训练模式：创建并拟合标准化器
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 保存标准化器
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        # 预测模式：加载已有标准化器
        if not os.path.exists(scaler_path):
            raise Exception("标准化器文件不存在，请先训练模型")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    
    # 第二步：可选PCA降维（用于消除特征间多重共线性，避免数值偏置）
    if pca_path is not None:
        if is_train:
            # 训练模式：创建并拟合PCA模型
            pca = PCA(n_components=n_components)
            X_processed = pca.fit_transform(X_scaled)
            # 保存PCA模型
            os.makedirs(os.path.dirname(pca_path), exist_ok=True)
            joblib.dump(pca, pca_path)
        else:
            # 预测模式：加载已有PCA模型
            if not os.path.exists(pca_path):
                raise Exception("PCA模型文件不存在，请先训练模型")
            pca = joblib.load(pca_path)
            X_processed = pca.transform(X_scaled)
        return X_processed
    
    # 不启用PCA时直接返回标准化后的数据
    return X_scaled
