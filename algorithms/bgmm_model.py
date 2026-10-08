import os
import joblib
from sklearn.mixture import BayesianGaussianMixture
from config import Config

def train_bgmm(X, n_components, model_name, dataset_id, max_iter=1000, tol=1e-5):
    """
    训练BGMM模型（scikit-learn 1.3.0兼容版），记录每一轮的损失值
    修复：n_init使用兼容的整数值，移除不支持的'auto'参数
    """
    # 初始化模型，先进行一次完整训练获取初始结果
    init_model = BayesianGaussianMixture(
        n_components=n_components,
        random_state=42,
        max_iter=max_iter,
        tol=tol,
        n_init=10,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1.0,
        warm_start=True,  # 开启热启动，支持逐轮训练
        verbose=0
    )
    
    loss_history = []
    prev_lower_bound = None
    # 每次只执行1次迭代，实现真正的增量训练
    init_model.max_iter = 1
    # 关闭tol检查，强制每次都执行一次迭代
    init_model.tol = 0
    
    for i in range(max_iter):
        # 单轮训练，每次只执行1次迭代
        init_model.fit(X)
        
        current_lower_bound = init_model.lower_bound_
        loss_history.append(-current_lower_bound)  # 转换为正损失值
        
        # 手动检查收敛
        if prev_lower_bound is not None and abs(current_lower_bound - prev_lower_bound) < tol:
            break
            
        prev_lower_bound = current_lower_bound
    
    # 保存最终模型
    model_dir = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{dataset_id}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(init_model, model_path)
    
    return init_model, model_path, loss_history
