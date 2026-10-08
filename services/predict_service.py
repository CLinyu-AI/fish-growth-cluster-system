import os
import json
import joblib
import pandas as pd
import numpy as np
from models import ModelInfo, PredictionRecord
from utils.preprocess import preprocess_data
from utils.database import db
from config import Config
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

# 加载鱼类名称中英文映射
def load_fish_name_map():
    name_map = {}
    name_file_path = os.path.join(Config.BASE_DIR, 'name.txt')
    if os.path.exists(name_file_path):
        with open(name_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # 跳过表头
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        chinese_name = parts[0].strip()
                        english_name = parts[1].strip()
                        name_map[english_name] = chinese_name
    return name_map

FISH_NAME_MAP = load_fish_name_map()

def get_deployed_models():
    """获取所有已部署的模型列表（包含准确率信息）"""
    from models import EvaluationMetric
    deployed_models = ModelInfo.query.filter_by(is_deployed=True).all()
    # 为每个模型补充准确率信息
    models_with_metric = []
    for model in deployed_models:
        metric = EvaluationMetric.query.filter_by(model_id=model.model_id).first()
        model.accuracy = metric.accuracy * 100 if metric else 0.0
        # 兼容数据库不存在remark字段的情况
        if not hasattr(model, 'remark'):
            model.remark = ''
        models_with_metric.append(model)
    return models_with_metric

def calculate_kmeans_confidence(model, X_new):
    """KMeans自定义置信度计算（基于到簇中心的距离）"""
    centers = model.cluster_centers_
    distances = np.linalg.norm(X_new - centers, axis=1)
    max_dist = np.max(distances)
    if max_dist == 0:
        return 100.0
    confidence = 100 * (1 - (distances[np.argmin(distances)] / max_dist))
    return round(confidence, 2)

def predict_fish(model_id, length, weight, w_l_ratio):
    """用指定模型预测鱼类类别"""
    # 加载模型基础信息
    model_info = ModelInfo.query.get(model_id)
    if not model_info or not model_info.is_deployed:
        return None, "模型不存在或未部署"
    
    try:
        # 1. 数据库中已经存储完整绝对路径，直接使用
        model_path = model_info.model_path
        scaler_path = model_info.scaler_path
        pca_path = getattr(model_info, 'pca_path', None)  # 兼容旧模型无pca_path字段
        
        if not os.path.exists(model_path):
            return None, f"模型文件不存在: {model_path}"
        
        # 修复scaler路径问题，自动查找scaler文件
        if not os.path.exists(scaler_path):
            # 1. 尝试在dataset目录下查找
            dataset_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', os.path.basename(scaler_path))
            if os.path.exists(dataset_scaler_path):
                scaler_path = dataset_scaler_path
            else:
                # 2. 尝试在saved_models根目录查找
                root_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, os.path.basename(scaler_path))
                if os.path.exists(root_scaler_path):
                    scaler_path = root_scaler_path
                else:
                    # 3. 尝试标准命名格式查找
                    standard_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, f'scaler_dataset_{model_info.dataset_id}.pkl')
                    if os.path.exists(standard_scaler_path):
                        scaler_path = standard_scaler_path
                    else:
                        return None, f"标准化器文件不存在，请重新训练模型"
        
        # 处理PCA路径问题，自动查找PCA模型文件
        if pca_path is not None and not os.path.exists(pca_path):
            # 尝试在dataset目录下查找
            dataset_pca_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', os.path.basename(pca_path))
            if os.path.exists(dataset_pca_path):
                pca_path = dataset_pca_path
            else:
                # 尝试在saved_models根目录查找
                root_pca_path = os.path.join(Config.SAVED_MODELS_DIR, os.path.basename(pca_path))
                if os.path.exists(root_pca_path):
                    pca_path = root_pca_path
        
        # 2. 加载模型
        model = joblib.load(model_path)
        algorithm_name = model_info.model_name
        # 2. 构造并预处理输入数据（标准化+可选PCA降维）
        input_data = pd.DataFrame([[length, weight, w_l_ratio]], columns=['length', 'weight', 'w_l_ratio'])
        X_scaled = preprocess_data(input_data, scaler_path, pca_path=pca_path, is_train=False)
        
        # 3. 分算法执行预测
        if algorithm_name == 'kmeans' or algorithm_name == 'kmeans++':
            pred_label = model.predict(X_scaled)[0]
            confidence = calculate_kmeans_confidence(model, X_scaled)
        elif algorithm_name == 'bgmm':
            pred_label = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            # 优化置信度计算：添加平滑防止极端值，避免一直显示100%
            max_proba = float(proba[pred_label])
            # 对概率进行轻微缩放，让置信度更合理，最高不超过99.9%
            confidence = round(min(max_proba * 100, 99.9), 2)
            # 如果所有簇的概率差异很小，适当降低置信度
            sorted_proba = sorted(proba, reverse=True)
            if len(sorted_proba) >= 2 and (sorted_proba[0] - sorted_proba[1]) < 0.1:
                confidence = round(confidence * 0.8, 2)
        elif algorithm_name == 'fcm':
            # FCM模型包含centers字段
            centers = model['centers']
            distances = np.linalg.norm(X_scaled - centers, axis=1)
            pred_label = np.argmin(distances)
            # 计算置信度
            max_dist = np.max(distances)
            if max_dist == 0:
                confidence = 100.0
            else:
                confidence = 100 * (1 - (distances[pred_label] / max_dist))
            confidence = round(confidence, 2)
        elif algorithm_name == 'meanshift':
            # MeanShift有predict方法
            pred_label = model.predict(X_scaled)[0]
            # 计算置信度
            centers = model.cluster_centers_
            distances = np.linalg.norm(X_scaled - centers, axis=1)
            max_dist = np.max(distances)
            if max_dist == 0:
                confidence = 100.0
            else:
                confidence = 100 * (1 - (distances[pred_label] / max_dist))
            confidence = round(confidence, 2)
        else:
            return None, "不支持的算法"
        
        # 4. 处理噪声标签
        if pred_label == -1:
            return {
                'predicted_label': int(pred_label),
                'predicted_species': '噪声样本（不属于任何已知类别）',
                'confidence': confidence,
                'model_name': algorithm_name.upper()
            }, None

        # 5. 加载物种名称映射
        map_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', f'model_{model_info.model_id}_species_map.json')
        species_map = {}
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                species_map = json.load(f)
        
        # 获取物种英文名并添加中文名称
        english_name = species_map.get(str(pred_label), f'未知类别{pred_label}')
        chinese_name = FISH_NAME_MAP.get(english_name, '')
        if chinese_name:
            display_name = f"{english_name}（{chinese_name}）"
        else:
            display_name = english_name
        
        return {
            'predicted_label': int(pred_label),
            'predicted_species': display_name,
            'confidence': confidence,
            'model_name': algorithm_name.upper()
        }, None
    except Exception as e:
        return None, f"预测失败：{str(e)}"

def batch_predict_fish(model_id, input_df):
    """
    批量预测鱼类类别
    :param model_id: 模型ID
    :param input_df: 输入DataFrame，必须包含length、weight、w_l_ratio列
    :return: 预测结果列表，错误信息
    """
    model_info = ModelInfo.query.get(model_id)
    if not model_info or not model_info.is_deployed:
        return None, "模型不存在或未部署"
    
    try:
        # 1. 加载模型和预处理工具
        model_path = model_info.model_path
        scaler_path = model_info.scaler_path
        pca_path = getattr(model_info, 'pca_path', None)
        
        if not os.path.exists(model_path):
            return None, f"模型文件不存在: {model_path}"
        
        # 处理scaler路径问题
        if not os.path.exists(scaler_path):
            # 1. 尝试在dataset目录下查找
            dataset_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', os.path.basename(scaler_path))
            if os.path.exists(dataset_scaler_path):
                scaler_path = dataset_scaler_path
            else:
                # 2. 尝试在saved_models根目录查找
                root_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, os.path.basename(scaler_path))
                if os.path.exists(root_scaler_path):
                    scaler_path = root_scaler_path
                else:
                    # 3. 尝试标准命名格式查找
                    standard_scaler_path = os.path.join(Config.SAVED_MODELS_DIR, f'scaler_dataset_{model_info.dataset_id}.pkl')
                    if os.path.exists(standard_scaler_path):
                        scaler_path = standard_scaler_path
                    else:
                        return None, "标准化器文件不存在，请重新训练模型"
        
        # 处理PCA路径问题
        if pca_path is not None and not os.path.exists(pca_path):
            dataset_pca_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', os.path.basename(pca_path))
            if os.path.exists(dataset_pca_path):
                pca_path = dataset_pca_path
            else:
                root_pca_path = os.path.join(Config.SAVED_MODELS_DIR, os.path.basename(pca_path))
                if os.path.exists(root_pca_path):
                    pca_path = root_pca_path
        
        # 2. 加载模型
        model = joblib.load(model_path)
        algorithm_name = model_info.model_name
        
        # 3. 批量预处理
        X_scaled = preprocess_data(input_df[['length', 'weight', 'w_l_ratio']], scaler_path, pca_path=pca_path, is_train=False)
        
        # 4. 批量预测
        results = []
        if algorithm_name == 'kmeans' or algorithm_name == 'kmeans++':
            pred_labels = model.predict(X_scaled)
            # 批量计算置信度
            centers = model.cluster_centers_
            for i, (x, pred_label) in enumerate(zip(X_scaled, pred_labels)):
                distances = np.linalg.norm(x - centers, axis=1)
                max_dist = np.max(distances)
                if max_dist == 0:
                    confidence = 100.0
                else:
                    confidence = 100 * (1 - (distances[pred_label] / max_dist))
                confidence = round(confidence, 2)
                results.append({
                    'index': i + 1,
                    'length': float(input_df.iloc[i]['length']),
                    'weight': float(input_df.iloc[i]['weight']),
                    'w_l_ratio': float(input_df.iloc[i]['w_l_ratio']),
                    'predicted_label': int(pred_label),
                    'confidence': confidence
                })
        elif algorithm_name == 'bgmm':
            pred_labels = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)
            for i, (pred_label, proba) in enumerate(zip(pred_labels, probas)):
                max_proba = float(proba[pred_label])
                confidence = round(min(max_proba * 100, 99.9), 2)
                sorted_proba = sorted(proba, reverse=True)
                if len(sorted_proba) >= 2 and (sorted_proba[0] - sorted_proba[1]) < 0.1:
                    confidence = round(confidence * 0.8, 2)
                results.append({
                    'index': i + 1,
                    'length': float(input_df.iloc[i]['length']),
                    'weight': float(input_df.iloc[i]['weight']),
                    'w_l_ratio': float(input_df.iloc[i]['w_l_ratio']),
                    'predicted_label': int(pred_label),
                    'confidence': confidence
                })
        elif algorithm_name == 'fcm':
            centers = model['centers']
            for i, x in enumerate(X_scaled):
                distances = np.linalg.norm(x - centers, axis=1)
                pred_label = np.argmin(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    confidence = 100.0
                else:
                    confidence = 100 * (1 - (distances[pred_label] / max_dist))
                confidence = round(confidence, 2)
                results.append({
                    'index': i + 1,
                    'length': float(input_df.iloc[i]['length']),
                    'weight': float(input_df.iloc[i]['weight']),
                    'w_l_ratio': float(input_df.iloc[i]['w_l_ratio']),
                    'predicted_label': int(pred_label),
                    'confidence': confidence
                })
        elif algorithm_name == 'meanshift':
            pred_labels = model.predict(X_scaled)
            centers = model.cluster_centers_
            for i, (x, pred_label) in enumerate(zip(X_scaled, pred_labels)):
                distances = np.linalg.norm(x - centers, axis=1)
                max_dist = np.max(distances)
                if max_dist == 0:
                    confidence = 100.0
                else:
                    confidence = 100 * (1 - (distances[pred_label] / max_dist))
                confidence = round(confidence, 2)
                results.append({
                    'index': i + 1,
                    'length': float(input_df.iloc[i]['length']),
                    'weight': float(input_df.iloc[i]['weight']),
                    'w_l_ratio': float(input_df.iloc[i]['w_l_ratio']),
                    'predicted_label': int(pred_label),
                    'confidence': confidence
                })
        else:
            return None, "不支持的算法"
        
        # 5. 加载物种名称映射
        map_path = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{model_info.dataset_id}', f'model_{model_info.model_id}_species_map.json')
        species_map = {}
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                species_map = json.load(f)
        
        # 添加物种名称
        for res in results:
            pred_label = res['predicted_label']
            if pred_label == -1:
                res['predicted_species'] = '噪声样本（不属于任何已知类别）'
            else:
                english_name = species_map.get(str(pred_label), f'未知类别{pred_label}')
                chinese_name = FISH_NAME_MAP.get(english_name, '')
                if chinese_name:
                    res['predicted_species'] = f"{english_name}（{chinese_name}）"
                else:
                    res['predicted_species'] = english_name
        
        # 6. 批量保存预测记录
        model_name = model_info.model_name
        user_id = None
        from flask import session
        if 'user_id' in session:
            user_id = session['user_id']
        
        if user_id:
            records = []
            for res in results:
                records.append(PredictionRecord(
                    user_id=user_id,
                    model_id=model_id,
                    model_name=model_name,
                    input_length=res['length'],
                    input_weight=res['weight'],
                    input_l_w_ratio=res['w_l_ratio'],
                    predicted_label=res['predicted_label'],
                    predicted_species=res['predicted_species'],
                    confidence=res['confidence']
                ))
            db.session.bulk_save_objects(records)
            db.session.commit()
        
        return results, None
    except Exception as e:
        db.session.rollback()
        return None, f"批量预测失败：{str(e)}"
