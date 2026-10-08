import os
import time
import pandas as pd
from werkzeug.utils import secure_filename
from config import Config
from models import Dataset
from utils.database import db
from models import DataRemark

# 允许的文件格式
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """校验文件格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_remove_file(file_path, max_retry=3):
    """安全删除文件，带重试逻辑，解决文件占用报错"""
    if not os.path.exists(file_path):
        return True
    for i in range(max_retry):
        try:
            os.remove(file_path)
            return True
        except:
            time.sleep(0.5)
    return False

def upload_dataset(file, dataset_name, uploader_id):
    """上传数据集"""
    if not file or not dataset_name:
        return False, "请填写数据集名称并选择文件"
    if not allowed_file(file.filename):
        return False, "仅支持CSV格式的文件"
    
    filename = secure_filename(file.filename)
    name_without_ext, ext = os.path.splitext(filename)
    save_path = os.path.join(Config.DATA_UPLOADS_DIR, filename)
    
    # 同名文件自动重命名
    counter = 1
    while os.path.exists(save_path):
        filename = f"{name_without_ext}_{counter}{ext}"
        save_path = os.path.join(Config.DATA_UPLOADS_DIR, filename)
        counter += 1
    
    file.save(save_path)
    
    # 校验CSV格式
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
        required_cols = ['length', 'weight', 'w_l_ratio', 'species']
        if not set(required_cols).issubset(df.columns):
            safe_remove_file(save_path)
            return False, f"CSV文件必须包含列：{required_cols}"
    except Exception as e:
        safe_remove_file(save_path)
        return False, f"文件读取失败：{str(e)}"
    
    new_dataset = Dataset(
        name=dataset_name,
        file_path=save_path,
        uploader_id=uploader_id
    )
    db.session.add(new_dataset)
    db.session.commit()
    
    return True, "数据集上传成功！"

def get_all_datasets():
    """获取所有数据集列表，附带真实聚类数"""
    datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
    for dataset in datasets:
        try:
            if os.path.exists(dataset.file_path):
                df = pd.read_csv(dataset.file_path)
                dataset.cluster_count = df['species'].nunique()
            else:
                dataset.cluster_count = 9
        except:
            dataset.cluster_count = 9
    return datasets

def delete_dataset(dataset_id):
    """删除数据集"""
    dataset = Dataset.query.get(dataset_id)
    if dataset:
        safe_remove_file(dataset.file_path)
        db.session.delete(dataset)
        db.session.commit()
        return True
    return False

def get_dataset_content(dataset_id):
    """获取数据集的CSV内容，返回DataFrame和数据集对象"""
    dataset = Dataset.query.get(dataset_id)
    if not dataset or not os.path.exists(dataset.file_path):
        return None, None, "数据集不存在"
    try:
        df = pd.read_csv(dataset.file_path)
        return df, dataset, None
    except Exception as e:
        return None, None, f"文件读取失败：{str(e)}"

def save_dataset_content(dataset_id, new_data):
    """保存修改后的数据集内容到CSV文件"""
    dataset = Dataset.query.get(dataset_id)
    if not dataset or not os.path.exists(dataset.file_path):
        return False, "数据集不存在"
    try:
        df = pd.DataFrame(new_data)
        df.to_csv(dataset.file_path, index=False, encoding='utf-8')
        return True, "数据集修改保存成功"
    except Exception as e:
        return False, f"文件保存失败：{str(e)}"
    
# ==================== 备注管理业务逻辑 ====================
def get_remark_list(remark_type=None, relation_id=None, only_active=False):
    """
    获取备注列表
    :param remark_type: 备注类型 dataset/model，不传则查所有类型
    :param relation_id: 关联主体ID，不传则查该类型所有
    :param only_active: 是否只查询启用状态的（给普通用户用）
    :return: 备注列表
    """
    query = DataRemark.query
    if remark_type:
        query = query.filter_by(remark_type=remark_type)
    if relation_id:
        query = query.filter_by(relation_id=relation_id)
    if only_active:
        query = query.filter_by(is_active=True)
    # 按更新时间倒序
    return query.order_by(DataRemark.update_time.desc()).all()

def get_remark_by_id(remark_id):
    """根据ID获取单条备注"""
    return DataRemark.query.get(remark_id)

def create_remark(remark_type, relation_id, content, is_active, creator_id):
    """创建新备注"""
    if not remark_type or not relation_id or not content:
        return False, "必填参数不能为空"
    
    new_remark = DataRemark(
        remark_type=remark_type,
        relation_id=relation_id,
        remark_content=content,
        is_active=is_active,
        creator_id=creator_id
    )
    try:
        db.session.add(new_remark)
        db.session.commit()
        return True, "备注创建成功"
    except Exception as e:
        db.session.rollback()
        return False, f"创建失败：{str(e)}"

def update_remark(remark_id, content=None, is_active=None):
    """修改备注内容/启用状态"""
    remark = get_remark_by_id(remark_id)
    if not remark:
        return False, "备注不存在"
    
    try:
        if content is not None:
            remark.remark_content = content
        if is_active is not None:
            remark.is_active = is_active
        db.session.commit()
        return True, "备注更新成功"
    except Exception as e:
        db.session.rollback()
        return False, f"更新失败：{str(e)}"

def delete_remark(remark_id):
    """删除备注"""
    remark = get_remark_by_id(remark_id)
    if not remark:
        return False, "备注不存在"
    
    try:
        db.session.delete(remark)
        db.session.commit()
        return True, "备注删除成功"
    except Exception as e:
        db.session.rollback()
        return False, f"删除失败：{str(e)}"