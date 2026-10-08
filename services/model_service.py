import os
import joblib
from models import ModelInfo, Dataset, EvaluationMetric
from utils.database import db
from sqlalchemy import desc

class ModelService:
    @staticmethod
    def get_model_by_id(model_id):
        """根据ID获取模型信息"""
        return ModelInfo.query.get(model_id)

    @staticmethod
    def get_all_models(dataset_id=None, page=1, page_size=20):
        """获取模型列表，支持按数据集筛选"""
        query = ModelInfo.query.order_by(desc(ModelInfo.create_time))
        if dataset_id:
            query = query.filter(ModelInfo.dataset_id == dataset_id)
        # 分页处理
        pagination = query.paginate(page=page, per_page=page_size, error_out=False)
        return pagination.items, pagination.total

    @staticmethod
    def load_model(model_id):
        """加载模型文件与标准化器，返回模型对象、scaler对象、模型信息"""
        model_info = ModelService.get_model_by_id(model_id)
        if not model_info:
            raise Exception(f"模型ID{model_id}不存在")
        
        # 校验模型文件存在
        if not os.path.exists(model_info.model_path):
            raise Exception("模型文件已丢失，请重新训练")
        if not os.path.exists(model_info.scaler_path):
            raise Exception("标准化器文件已丢失，请重新训练")
        
        # 加载模型与scaler
        model = joblib.load(model_info.model_path)
        scaler = joblib.load(model_info.scaler_path)
        
        return model, scaler, model_info

    @staticmethod
    def deploy_model(model_id):
        """部署模型（上线到前端可用）"""
        model_info = ModelService.get_model_by_id(model_id)
        if not model_info:
            raise Exception(f"模型ID{model_id}不存在")
        
        # 先下线同数据集的其他已部署模型，保证单数据集单部署模型
        ModelInfo.query.filter(
            ModelInfo.dataset_id == model_info.dataset_id,
            ModelInfo.is_deployed == True
        ).update({ModelInfo.is_deployed: False})
        
        # 部署当前模型
        model_info.is_deployed = True
        db.session.commit()
        return True

    @staticmethod
    def offline_model(model_id):
        """下线模型"""
        model_info = ModelService.get_model_by_id(model_id)
        if not model_info:
            raise Exception(f"模型ID{model_id}不存在")
        
        model_info.is_deployed = False
        db.session.commit()
        return True

    @staticmethod
    def get_model_detail(model_id):
        """获取模型完整详情（含数据集、评估指标）"""
        model_info = ModelService.get_model_by_id(model_id)
        if not model_info:
            raise Exception(f"模型ID{model_id}不存在")
        
        # 关联数据集与指标
        dataset = Dataset.query.get(model_info.dataset_id)
        metric = EvaluationMetric.query.filter(EvaluationMetric.model_id == model_id).first()
        
        return {
            "model_info": model_info,
            "dataset": dataset,
            "metric": metric
        }

    @staticmethod
    def delete_model(model_id):
        """删除模型（含数据库记录与本地文件）"""
        model_info = ModelService.get_model_by_id(model_id)
        if not model_info:
            raise Exception(f"模型ID{model_id}不存在")
        
        # 禁止删除已部署模型
        if model_info.is_deployed:
            raise Exception("已部署的模型无法删除，请先下线")
        
        # 删除本地文件
        try:
            if os.path.exists(model_info.model_path):
                os.remove(model_info.model_path)
            # 不删除scaler，同数据集其他模型可能复用
        except Exception as e:
            raise Exception(f"模型文件删除失败：{str(e)}")
        
        # 删除数据库记录
        EvaluationMetric.query.filter(EvaluationMetric.model_id == model_id).delete()
        db.session.delete(model_info)
        db.session.commit()
        
        return True

    @staticmethod
    def get_deployed_model(dataset_id):
        """获取指定数据集已部署的模型"""
        return ModelInfo.query.filter(
            ModelInfo.dataset_id == dataset_id,
            ModelInfo.is_deployed == True
        ).first()

# 单例导出
model_service = ModelService()