from sklearn.cluster import KMeans
import joblib
import os
import numpy as np
from config import Config

def train_kmeans(X, n_clusters, model_name, dataset_id, max_iter=300, tol=1e-4):
    """
    自定义KMeans训练，记录每一轮的损失值（兼容低版本sklearn，无warm_start）
    """
    loss_history = []
    prev_inertia = None
    current_centers = 'k-means++'
    
    for i in range(max_iter):
        # 每轮新建模型，基于上一轮中心点继续训练
        model = KMeans(
            n_clusters=n_clusters,
            init=current_centers,
            random_state=42,
            n_init=1 if i > 0 else 'auto',
            max_iter=1,
            tol=tol,
            verbose=0
        )
        model.fit(X)
        
        current_inertia = model.inertia_
        loss_history.append(current_inertia)
        current_centers = model.cluster_centers_
        
        # 检查收敛
        if prev_inertia is not None and abs(prev_inertia - current_inertia) < tol:
            break
            
        prev_inertia = current_inertia
    
    # 最后训练完整模型确保最优结果
    final_model = KMeans(
        n_clusters=n_clusters,
        init=current_centers,
        random_state=42,
        n_init=1,
        max_iter=max_iter,
        tol=tol
    )
    final_model.fit(X)
    
    model_dir = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{dataset_id}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(final_model, model_path)
    
    return final_model, model_path, loss_history
