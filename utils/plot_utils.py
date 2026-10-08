# 必须在import pyplot之前设置后端，禁用GUI，消除线程警告
import matplotlib
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from config import Config
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KernelDensity

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_clustering_metrics(X, y_pred):
    """计算无监督聚类核心指标"""
    valid_mask = y_pred != -1
    X_valid = X[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    n_clusters = len(np.unique(y_pred_valid))
    if n_clusters < 2:
        return {
            'silhouette_score': 0.0,
            'dbi_score': 9999.99,
            'ch_score': 0.0
        }
    
    return {
        'silhouette_score': round(silhouette_score(X_valid, y_pred_valid), 4),
        'dbi_score': round(davies_bouldin_score(X_valid, y_pred_valid), 4),
        'ch_score': round(calinski_harabasz_score(X_valid, y_pred_valid), 4)
    }

def generate_model_plots(model_id, X_scaled, y_pred, y_true, accuracy, algorithm_name, model=None, loss_history=None):
    """
    仅支持4种新算法的可视化生成：
    1. 通用：聚类结果散点图（PCA降维）
    2. 通用：准确率曲线
    3. 通用：无监督指标雷达图
    4. 专属：各算法特色图
    5. 专属：损失迭代曲线（仅KMeans/BGMM/FCM）
    """
    save_dir = os.path.join(Config.BASE_DIR, 'static', 'model_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 1. 通用：聚类结果散点图（PCA降维） ==========
    plt.figure(figsize=(6, 4))  # 缩小尺寸
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7, s=30)
    
    # 标注聚类中心（如果模型有）
    if algorithm_name == 'kmeans' and hasattr(model, 'cluster_centers_'):
        centers_2d = pca.transform(model.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, linewidths=1.5, label='聚类中心')
    elif algorithm_name == 'meanshift' and hasattr(model, 'cluster_centers_'):
        centers_2d = pca.transform(model.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, linewidths=1.5, label='聚类中心')
    elif algorithm_name == 'fcm' and 'centers' in model:
        centers_2d = pca.transform(model['centers'])
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, linewidths=1.5, label='聚类中心')
    
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'类别{i}' for i in np.unique(y_pred)], title='聚类结果', loc='best', fontsize=8)
    plt.title(f'{algorithm_name.upper()} 聚类结果散点图', fontsize=10)
    plt.xlabel('PCA主成分1', fontsize=8)
    plt.ylabel('PCA主成分2', fontsize=8)
    plt.tick_params(axis='both', labelsize=7)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'model_{model_id}_scatter.png'), dpi=150, bbox_inches='tight')  # 降低dpi
    plt.close()
    
    # ========== 2. 通用：ARI对比 ==========
    plt.figure(figsize=(5, 4))  # 缩小尺寸
    unique_classes = len(np.unique(y_true))
    random_acc = 100 / unique_classes if unique_classes > 0 else 10
    
    # 绘制ARI对比
    bars = plt.bar(['模型ARI', '随机基准ARI'], [accuracy*100, random_acc], color=['#27ae60', '#e74c3c'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+1, f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.title('模型ARI对比', fontsize=10)
    plt.ylabel('ARI（%）', fontsize=8)
    plt.ylim(0, 100)
    plt.tick_params(axis='both', labelsize=7)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'model_{model_id}_accuracy.png'), dpi=150, bbox_inches='tight')  # 降低dpi
    plt.close()
    
    # ========== 3. 通用：无监督指标雷达图 ==========
    metrics = calculate_clustering_metrics(X_scaled, y_pred)
    # 指标归一化到0-1
    silhouette_norm = max(0, min(1, metrics['silhouette_score']))
    dbi_norm = max(0, min(1, 1 - metrics['dbi_score']/10))  # DBI越小越好，反向归一化
    ch_norm = max(0, min(1, metrics['ch_score']/100))  # CH越大越好，缩放
    
    categories = ['轮廓系数', 'DBI指数', 'CH指数']
    values = [silhouette_norm, dbi_norm, ch_norm]
    
    plt.figure(figsize=(5, 5))  # 缩小尺寸
    ax = plt.subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, color='#3498db', linewidth=1.5, linestyle='solid')
    ax.fill(angles, values, color='#3498db', alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_title(f'{algorithm_name.upper()} 无监督评估指标', fontsize=10, y=1.1)
    
    # 标注原始值
    for i, (angle, value, raw) in enumerate(zip(angles[:-1], values[:-1], [metrics['silhouette_score'], metrics['dbi_score'], metrics['ch_score']])):
        ax.text(angle, value + 0.05, f'{raw}', ha='center', va='bottom', fontsize=7)
    
    plt.savefig(os.path.join(save_dir, f'model_{model_id}_radar.png'), dpi=150, bbox_inches='tight')  # 降低dpi
    plt.close()
    
    # ========== 4. 专属：各算法特色图 ==========
    if algorithm_name == 'kmeans':
        # K-Means++ 特色：损失迭代曲线
        if loss_history and len(loss_history) > 0:
            plt.figure(figsize=(6, 4))  # 缩小尺寸
            plt.plot(range(1, len(loss_history)+1), loss_history, color='#e74c3c', linewidth=1.5, marker='o', markersize=3)
            plt.title('K-Means++ 损失迭代曲线', fontsize=10)
            plt.xlabel('迭代轮次', fontsize=8)
            plt.ylabel('Inertia', fontsize=8)
            plt.tick_params(axis='both', labelsize=7)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'model_{model_id}_loss_curve.png'), dpi=150, bbox_inches='tight')  # 降低dpi
            plt.close()
    
    elif algorithm_name == 'bgmm':
        # BGMM 特色：高斯分量权重分布 + 损失迭代曲线
        if model and hasattr(model, 'weights_'):
            plt.figure(figsize=(6, 4))  # 缩小尺寸
            weights = model.weights_
            plt.bar(range(1, len(weights)+1), weights, color='#9b59b6', alpha=0.7)
            plt.title('BGMM 高斯分量权重分布', fontsize=10)
            plt.xlabel('分量编号', fontsize=8)
            plt.ylabel('权重', fontsize=8)
            plt.tick_params(axis='both', labelsize=7)
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'model_{model_id}_weights.png'), dpi=150, bbox_inches='tight')  # 降低dpi
            plt.close()
        
        if loss_history and len(loss_history) > 0:
            plt.figure(figsize=(6, 4))  # 缩小尺寸
            plt.plot(range(1, len(loss_history)+1), loss_history, color='#e74c3c', linewidth=1.5, marker='o', markersize=3)
            plt.title('BGMM 损失迭代曲线', fontsize=10)
            plt.xlabel('迭代轮次', fontsize=8)
            plt.ylabel('负对数似然', fontsize=8)
            plt.tick_params(axis='both', labelsize=7)
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'model_{model_id}_loss_curve.png'), dpi=150, bbox_inches='tight')  # 降低dpi
            plt.close()
    
    # ========== FCM 特色：隶属度矩阵热力图 + 损失迭代曲线 ==========
    elif algorithm_name == 'fcm':
        # 强制初始化保存路径
        save_dir = os.path.join(Config.BASE_DIR, 'static', 'model_plots')
        os.makedirs(save_dir, exist_ok=True)
        print(f"[FCM热力图] 保存目录：{save_dir}")

        # 1. 隶属度矩阵热力图（双方案兜底版）
        if model and isinstance(model, dict) and 'membership' in model:
            heatmap_path = os.path.join(save_dir, f'model_{model_id}_membership.png')
            membership = model['membership']
            print(f"[FCM热力图] 隶属度矩阵形状：{membership.shape}")

            try:
                # 强制固定画布尺寸，避免异常
                plt.figure(figsize=(6, 3), dpi=150)
                
                # 样本容错：最少5个，最多20个，避免图过大/生成失败
                total_samples = membership.shape[1]
                sample_size = min(20, total_samples)
                if total_samples > sample_size:
                    membership_sample = membership[:, :sample_size]
                    print(f"[FCM热力图] 样本数过多，截取前{sample_size}个样本")
                else:
                    membership_sample = membership

                # 优先用seaborn，失败则用matplotlib原生兜底
                try:
                    import seaborn as sns
                    print("[FCM热力图] 使用seaborn生成热力图")
                    sns.heatmap(
                        membership_sample, 
                        cmap='viridis', 
                        xticklabels=range(1, sample_size+1), 
                        yticklabels=[f'类别{i}' for i in range(membership_sample.shape[0])],
                        annot=False,
                        cbar=True,
                        cbar_kws={"shrink": 0.7}
                    )
                except ImportError:
                    print("[FCM热力图] 未安装seaborn，使用matplotlib原生方案")
                    # matplotlib原生热力图兜底
                    plt.imshow(membership_sample, cmap='viridis', aspect='auto')
                    plt.colorbar(label='隶属度', shrink=0.7)
                    # 设置坐标轴
                    plt.xticks(range(sample_size), range(1, sample_size+1), fontsize=6)
                    plt.yticks(range(membership_sample.shape[0]), [f'类别{i}' for i in range(membership_sample.shape[0])], fontsize=6)

                # 统一设置标题和标签
                plt.title('FCM 隶属度矩阵热力图', fontsize=9)
                plt.xlabel('样本编号', fontsize=7)
                plt.ylabel('聚类类别', fontsize=7)
                plt.tick_params(axis='both', labelsize=6)
                
                # 强制保存，覆盖已有文件
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                print(f"[FCM热力图] 生成成功！文件路径：{heatmap_path}")
                print(f"[FCM热力图] 文件是否存在：{os.path.exists(heatmap_path)}")

            except Exception as e:
                print(f"[FCM热力图] 生成失败，错误原因：{str(e)}")
            finally:
                # 强制关闭画布，释放资源
                plt.close('all')
        else:
            print(f"[FCM热力图] 模型结构异常，缺少membership字段，模型内容：{model.keys() if isinstance(model, dict) else model}")
        
        # 2. 损失迭代曲线（固定尺寸+强校验）
        if model and isinstance(model, dict) and 'loss_history' in model:
            loss_curve_path = os.path.join(save_dir, f'model_{model_id}_loss_curve.png')
            loss_hist = model['loss_history']
            print(f"[FCM损失曲线] 损失历史长度：{len(loss_hist)}")

            try:
                plt.figure(figsize=(6, 3), dpi=150)
                plt.plot(range(1, len(loss_hist)+1), loss_hist, color='#e74c3c', linewidth=1.2, marker='o', markersize=2)
                plt.title('FCM 目标函数迭代曲线', fontsize=9)
                plt.xlabel('迭代轮次', fontsize=7)
                plt.ylabel('目标函数值', fontsize=7)
                plt.tick_params(axis='both', labelsize=6)
                plt.grid(alpha=0.3)
                plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                print(f"[FCM损失曲线] 生成成功！文件路径：{loss_curve_path}")
            except Exception as e:
                print(f"[FCM损失曲线] 生成失败，错误原因：{str(e)}")
            finally:
                plt.close('all')
    
    elif algorithm_name == 'meanshift':
        # MeanShift 特色：核密度估计图
        plt.figure(figsize=(6, 4))  # 缩小尺寸
        # PCA降维到1维做KDE
        pca_1d = PCA(n_components=1, random_state=42)
        X_1d = pca_1d.fit_transform(X_scaled)
        
        # 核密度估计
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(X_1d)
        x_range = np.linspace(X_1d.min()-1, X_1d.max()+1, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x_range)
        
        plt.fill(x_range, np.exp(log_dens), color='#3498db', alpha=0.5, label='核密度估计')
        plt.scatter(X_1d, np.zeros_like(X_1d), c=y_pred, cmap='viridis', s=20, alpha=0.7, label='样本点')
        
        # 标注聚类中心
        if hasattr(model, 'cluster_centers_'):
            centers_1d = pca_1d.transform(model.cluster_centers_)
            plt.scatter(centers_1d, np.zeros_like(centers_1d), c='red', marker='X', s=100, linewidths=1.5, label='聚类中心')
        
        plt.title('MeanShift 核密度估计分布', fontsize=10)
        plt.xlabel('PCA降维特征', fontsize=8)
        plt.ylabel('密度', fontsize=8)
        plt.legend(fontsize=7)
        plt.tick_params(axis='both', labelsize=7)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'model_{model_id}_kde.png'), dpi=150, bbox_inches='tight')  # 降低dpi
        plt.close()
    
    return True