import joblib

def load_model(model_path):
    """加载训练好的模型"""
    return joblib.load(model_path)