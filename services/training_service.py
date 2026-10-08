import os
import joblib
import pandas as pd
import json
import numpy as np
import skfuzzy as fuzz
from config import Config
from utils.preprocess import preprocess_data
from utils.metrics import calculate_ari, calculate_clustering_loss, calculate_clustering_metrics
from scipy import stats
# 修复1：新增导入auto_select_params统一调参入口
from utils.auto_tune import (
    auto_select_params,
    auto_select_kmeans_params, 
    auto_select_bgmm_params, 
    auto_select_fcm_params, 
    auto_select_meanshift_params
)
from utils.plot_utils import generate_model_plots
from algorithms.kmeans_model import train_kmeans
from algorithms.bgmm_model import train_bgmm
from algorithms.fcm_model import train_fcm
from algorithms.meanshift_model import train_meanshift
from models import Dataset, ModelInfo, EvaluationMetric, TrainingTask
from utils.database import db

def run_training(dataset_id, algorithm_name, n_clusters=None):
    """
    核心训练流程（仅支持4种新算法）
    支持: kmeans, bgmm, fcm, meanshift
    """
    # 1. 加载数据
    dataset = Dataset.query.get(dataset_id)
    if not dataset: 
        raise Exception(f"数据集ID{dataset_id}不存在")
    df = pd.read_csv(dataset.file_path)
    
    # 2. 预处理
    X = df[['length', 'weight', 'w_l_ratio']]
    y_true = df['species'].astype('category').cat.codes
    n_true_clusters = df['species'].nunique()
    
    scaler_path = os.path.join(Config.SAVED_MODELS_DIR, f'scaler_dataset_{dataset_id}.pkl')
    pca_path = os.path.join(Config.SAVED_MODELS_DIR, f'pca_dataset_{dataset_id}.pkl')
    # 启用PCA降维消除特征多重共线性，保留95%方差信息
    X_scaled = preprocess_data(X, scaler_path, pca_path=pca_path, is_train=True, n_components=0.95)
    
    # 3. 创建训练任务
    task = TrainingTask(dataset_id=dataset_id, algorithm=algorithm_name, status='running')
    db.session.add(task)
    db.session.commit()
    
    # 4. 训练与自动调参
    model_name = f'{algorithm_name}_dataset{dataset_id}_task{task.task_id}'
    model, model_path, loss_history, final_n = None, None, None, n_clusters
    
    try:
        # 修复2：统一使用auto_select_params获取最优超参数，所有算法逻辑对齐
        best_params = auto_select_params(X_scaled, algorithm_name, n_true_clusters)
        
        # ========== K-Means++ 算法 ==========
        if algorithm_name == 'kmeans':
            final_k = n_clusters or best_params['n_clusters']
            model, model_path, loss_history = train_kmeans(X_scaled, final_k, model_name, dataset_id)
            final_n = final_k
        
        # ========== BGMM 算法 ==========
        elif algorithm_name == 'bgmm':
            # 修复3：统一使用best_params，不再混用单独调参函数
            final_k = n_clusters or best_params['n_clusters']
            model, model_path, loss_history = train_bgmm(
                X=X_scaled,
                n_components=final_k,
                model_name=model_name,
                dataset_id=dataset_id
            )
            final_n = final_k  # 修复4：变量名统一，避免未定义
            
        # ========== FCM 模糊C均值聚类 ==========
        elif algorithm_name == 'fcm':
            final_k = n_clusters or best_params['n_clusters']
            model, model_path = train_fcm(X_scaled, final_k, model_name, dataset_id, m=best_params['m'])
            loss_history = model['loss_history'].tolist()
            final_n = final_k
            
        # ========== MeanShift 均值漂移 ==========
        elif algorithm_name == 'meanshift':
            model, model_path = train_meanshift(X_scaled, model_name, dataset_id, bandwidth=best_params['bandwidth'])
            final_n = len(np.unique(model.labels_))
            loss_history = None  # MeanShift无迭代损失
            
        # ========== 不支持的算法 ==========
        else:
            raise Exception(f"不支持算法：{algorithm_name}，支持算法：kmeans、bgmm、fcm、meanshift")
    
    except Exception as e:
        # 修复：异常时主动回滚数据库会话，避免PendingRollbackError
        db.session.rollback()
        task.status = 'failed'
        db.session.commit()
        raise e
    
    # ========== 新增：使用增广数据集进行模型二次修正 ==========
    def load_augmented_datasets():
        """加载ultidataset中的五个增广数据集"""
        aug_dir = os.path.join(Config.BASE_DIR, 'ultidataset')
        aug_dfs = []
        if os.path.exists(aug_dir):
            for file in os.listdir(aug_dir):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(aug_dir, file))
                    if all(col in df.columns for col in ['length', 'weight', 'w_l_ratio', 'species']):
                        aug_dfs.append(df)
        return aug_dfs
    
    # 加载增广数据集
    augmented_dfs = load_augmented_datasets()
    std_metrics = {}
    ablation_results = []
    
    if augmented_dfs and len(augmented_dfs) >= 5:
        # 收集所有增广数据
        all_aug_X = []
        all_aug_y = []
        for aug_df in augmented_dfs:
            aug_X = aug_df[['length', 'weight', 'w_l_ratio']]
            aug_y = aug_df['species'].astype('category').cat.codes
            # 使用已有的scaler和PCA进行预处理
            aug_X_scaled = preprocess_data(aug_X, scaler_path, pca_path=pca_path, is_train=False)
            all_aug_X.append(aug_X_scaled)
            all_aug_y.append(aug_y)
        
        # 模型二次修正（微调）
        if algorithm_name == 'kmeans':
            # KMeans微调：合并原始数据和增广数据重新训练
            combined_X = np.vstack([X_scaled] + all_aug_X)
            model, model_path, loss_history = train_kmeans(combined_X, final_n, model_name, dataset_id)
        elif algorithm_name == 'bgmm':
            # BGMM微调：使用增广数据重新拟合
            combined_X = np.vstack([X_scaled] + all_aug_X)
            model, model_path, loss_history = train_bgmm(
                X=combined_X,
                n_components=final_n,
                model_name=model_name,
                dataset_id=dataset_id
            )
        elif algorithm_name == 'fcm':
            # FCM微调：合并数据重新训练
            combined_X = np.vstack([X_scaled] + all_aug_X)
            model, model_path = train_fcm(combined_X, final_n, model_name, dataset_id, m=best_params['m'])
            loss_history = model['loss_history'].tolist()
        
        # 计算标准差：在每个增广数据集上评估，计算指标的标准差
        metric_list = []
        for i, (aug_X, aug_y) in enumerate(zip(all_aug_X, all_aug_y)):
            if algorithm_name == 'fcm':
                u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(aug_X.T, model['centers'], m=best_params['m'], error=1e-5, maxiter=1000)
                aug_pred = np.argmax(u, axis=0)
            else:
                aug_pred = model.predict(aug_X)
            
            acc = calculate_ari(aug_y, aug_pred)
            met = calculate_clustering_metrics(aug_X, aug_pred)
            metric_list.append({
                'ari': acc,
                'silhouette': met['silhouette_score'],
                'dbi': met['dbi_score'],
                'ch': met['ch_score']
            })
            
            # 记录消融实验数据
            ablation_results.append({
                'dataset_index': i,
                'algorithm': algorithm_name,
                'n_clusters': final_n,
                'ari': acc,
                'silhouette_score': met['silhouette_score'],
                'dbi_score': met['dbi_score'],
                'ch_score': met['ch_score']
            })
        
        # 计算各指标的标准差
        if metric_list:
            acc_values = [m['ari'] for m in metric_list]
            sil_values = [m['silhouette'] for m in metric_list]
            dbi_values = [m['dbi'] for m in metric_list]
            ch_values = [m['ch'] for m in metric_list]
            
            std_metrics = {
                'ari_std': float(np.std(acc_values)),
                'silhouette_std': float(np.std(sil_values)),
                'dbi_std': float(np.std(dbi_values)),
                'ch_std': float(np.std(ch_values)),
                'sample_count': len(metric_list)
            }
            
            # 保存标准差和消融实验结果
            result_dir = os.path.join(Config.SAVED_MODELS_DIR, f'experiment_results')
            os.makedirs(result_dir, exist_ok=True)
            
            # 先暂存结果，等model_info创建后再保存
            temp_std_metrics = std_metrics
            temp_ablation_results = ablation_results
    else:
        temp_std_metrics = None
        temp_ablation_results = None
    
    # 5. 保存模型信息到数据库
    display_name = algorithm_name
    if algorithm_name == 'kmeans':
        display_name = 'kmeans++'
    model_info = ModelInfo(
        model_name=display_name, 
        model_path=model_path, 
        scaler_path=scaler_path,
        pca_path=pca_path, 
        dataset_id=dataset_id, 
        is_deployed=False
    )
    db.session.add(model_info)
    db.session.commit()
    
    # 6. 统一获取聚类标签，计算评估指标（模型已用合并数据训练，需单独预测原始数据集的标签）
    if algorithm_name == 'fcm':
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(X_scaled.T, model['centers'], m=best_params['m'], error=1e-5, maxiter=1000)
        y_pred = np.argmax(u, axis=0)
    else:
        y_pred = model.predict(X_scaled)
    
    ari = calculate_ari(y_true, y_pred)
    loss_value = calculate_clustering_loss(algorithm_name, model, X_scaled)
    metrics = calculate_clustering_metrics(X_scaled, y_pred)
    
    # 保存评估指标
    eval_metric_args = {
        'model_id': model_info.model_id,
        'accuracy': round(ari, 4),
        'loss_value': round(loss_value, 4),
        'silhouette_score': metrics['silhouette_score'],
        'dbi_score': metrics['dbi_score'],
        'ch_score': metrics['ch_score']
    }
    # 有标准差数据则保存
    if temp_std_metrics:
        eval_metric_args.update({
            'accuracy_std': round(temp_std_metrics['ari_std'], 4),
            'silhouette_std': round(temp_std_metrics['silhouette_std'], 4),
            'dbi_std': round(temp_std_metrics['dbi_std'], 4),
            'ch_std': round(temp_std_metrics['ch_std'], 4)
        })
    db.session.add(EvaluationMetric(**eval_metric_args))
    
    # 保存之前暂存的标准差和消融实验结果
    if temp_std_metrics and temp_ablation_results:
        result_dir = os.path.join(Config.SAVED_MODELS_DIR, f'experiment_results')
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存标准差结果
        std_path = os.path.join(result_dir, f'model_{model_info.model_id}_std_metrics.json')
        with open(std_path, 'w', encoding='utf-8') as f:
            json.dump(temp_std_metrics, f, ensure_ascii=False, indent=2)
        
        # 保存消融实验结果
        ablation_path = os.path.join(result_dir, f'model_{model_info.model_id}_ablation_results.json')
        with open(ablation_path, 'w', encoding='utf-8') as f:
            json.dump(temp_ablation_results, f, ensure_ascii=False, indent=2)
    
    # 生成并保存物种名称映射文件：聚类标签 -> 真实物种名
    from collections import Counter
    species_categories = df['species'].astype('category').cat.categories
    # 为每个聚类标签匹配出现最多的真实物种
    cluster_to_species = {}
    unique_clusters = np.unique(y_pred)
    for cluster in unique_clusters:
        # 找到该聚类中所有样本的真实物种
        true_labels_in_cluster = y_true[y_pred == cluster]
        # 统计出现次数最多的真实物种
        most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
        cluster_to_species[str(cluster)] = species_categories[most_common_label]
    
    # 创建保存目录
    map_dir = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{dataset_id}')
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, f'model_{model_info.model_id}_species_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_to_species, f, ensure_ascii=False, indent=2)
    
    # 7. 生成可视化图表（修复5：删除重复调用，只保留一次完整调用）
    generate_model_plots(
        model_id=model_info.model_id,
        X_scaled=X_scaled,
        y_pred=y_pred,
        y_true=y_true,
        accuracy=ari,
        algorithm_name=algorithm_name,
        model=model,
        loss_history=loss_history
    )
    
    # 8. 更新训练任务状态
    task.status = 'completed'
    task.final_n_clusters = final_n
    db.session.commit()
    
    return model_info