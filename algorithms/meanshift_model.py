import os
import joblib
from sklearn.cluster import MeanShift, estimate_bandwidth
from config import Config

def train_meanshift(X, model_name, dataset_id, bandwidth=None):
    if bandwidth is None:
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=42)
        bandwidth = max(bandwidth, 0.1)

    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    model.fit(X)

    model_dir = os.path.join(Config.SAVED_MODELS_DIR, f'dataset_{dataset_id}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)

    return model, model_path