import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score

def calculate_ari(y_true, y_pred):
    valid_mask = y_pred != -1
    if np.sum(valid_mask) < 2: return 0.0
    return max(adjusted_rand_score(y_true[valid_mask], y_pred[valid_mask]), 0.0)

def calculate_clustering_metrics(X, y_pred):
    valid_mask = y_pred != -1
    X_valid, y_pred_valid = X[valid_mask], y_pred[valid_mask]
    if len(np.unique(y_pred_valid)) < 2:
        return {'silhouette_score': 0.0, 'dbi_score': 9999.99, 'ch_score': 0.0}
    
    return {
        'silhouette_score': round(silhouette_score(X_valid, y_pred_valid), 4),
        'dbi_score': round(davies_bouldin_score(X_valid, y_pred_valid), 4),
        'ch_score': round(calinski_harabasz_score(X_valid, y_pred_valid), 4)
    }

def calculate_clustering_loss(algorithm_name, model, X):
    if algorithm_name == 'kmeans':
        return model.inertia_
    elif algorithm_name == 'bgmm':
        return -model.score(X)
    elif algorithm_name == 'fcm':
        return model['loss_history'][-1]
    elif algorithm_name == 'meanshift':
        labels = model.labels_
        if len(np.unique(labels[labels != -1])) < 2: return 9999.99
        return 1 / (silhouette_score(X[labels != -1], labels[labels != -1]) + 1e-6)
    return 0.0