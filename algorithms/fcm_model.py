import os
import joblib
import numpy as np
import skfuzzy as fuzz
from config import Config

def train_fcm(X, n_clusters, model_name, dataset_id, m=2.0):
    """
    训练模糊C均值聚类FCM模型（最终稳定版）
    """
    # FCM要求输入为(特征数, 样本数)格式
    X_T = X.T
    # 执行FCM聚类
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_T,
        c=n_clusters,
        m=m,
        error=1e-5,
        maxiter=1000,
        seed=42,
        init=None
    )
    print(f"[FCM训练] 聚类完成，迭代次数：{p}，最终损失：{jm[-1]}")

    # 固定模型结构，确保字段齐全
    fcm_model = {
        'centers': cntr,
        'm': m,
        'n_clusters': n_clusters,
        'membership': u,  # 隶属度矩阵，热力图必需
        'loss_history': jm,  # 迭代损失历史，损失曲线必需
        'fpc': fpc,
        'labels': np.argmax(u, axis=0)  # 硬聚类标签
    }

    # 保存模型
    model_dir = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{dataset_id}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(fcm_model, model_path)
    print(f"[FCM训练] 模型保存成功：{model_path}")

    return fcm_model, model_path

# FCM预测函数
def predict_fcm(model, X):
    X_T = X.T
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        X_T, model['centers'], m=model['m'], error=1e-5, maxiter=1000, seed=42
    )
    labels = np.argmax(u, axis=0)
    return labels, u