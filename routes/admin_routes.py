from flask import Blueprint, render_template, request, redirect, url_for, session
from config import Config
from services.training_service import run_training
from services.data_service import upload_dataset, get_all_datasets, delete_dataset, get_dataset_content, save_dataset_content
from services.user_manage_service import get_all_users, create_user, delete_user, update_user_role
from models import ModelInfo, Dataset, UserRole, TrainingTask, EvaluationMetric, AdminOperationLog, User
from utils.database import db
import os
from services.data_service import get_remark_list, create_remark, update_remark, delete_remark
from utils.log_utils import add_admin_log

admin_bp = Blueprint('admin', __name__)

# 1. 管理员首页
@admin_bp.route('/')
def dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    # 统计数据
    user_count = get_all_users().__len__()
    dataset_count = get_all_datasets().__len__()
    model_count = ModelInfo.query.count()
    task_count = TrainingTask.query.count()
    return render_template('admin/dashboard.html', 
                           active_page='dashboard',
                           user_count=user_count,
                           dataset_count=dataset_count,
                           model_count=model_count,
                           task_count=task_count)

# 2. 数据管理页面
@admin_bp.route('/data', methods=['GET', 'POST'])
def data_manage():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        # 上传数据集
        file = request.files.get('dataset_file')
        dataset_name = request.form.get('dataset_name')
        uploader_id = session['user_id']
        success, msg = upload_dataset(file, dataset_name, uploader_id)
        if success:
            return render_template('admin/data.html', active_page='data', datasets=get_all_datasets(), success=msg)
        else:
            return render_template('admin/data.html', active_page='data', datasets=get_all_datasets(), error=msg)
    
    # GET请求，展示数据集列表
    datasets = get_all_datasets()
    return render_template('admin/data.html', active_page='data', datasets=datasets)

# 3. 删除数据集
@admin_bp.route('/data/delete/<int:dataset_id>')
def delete_dataset_route(dataset_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    delete_dataset(dataset_id)
    add_admin_log(f"删除数据集ID: {dataset_id}")
    return redirect(url_for('admin.data_manage'))

# 4. 模型训练页面
# 模型训练页面
# 模型训练页面（GET：渲染页面）
@admin_bp.route('/train', methods=['GET'])
def train_manage():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    # 算法列表：支持kmeans、bgmm、fcm、meanshift
    algorithms = ['kmeans', 'bgmm', 'fcm', 'meanshift']
    datasets = get_all_datasets()
    return render_template('admin/train.html', active_page='train', algorithms=algorithms, datasets=datasets)

# 提交训练任务（POST：处理训练请求，核心修复methods）
@admin_bp.route('/train/run', methods=['POST'])
def run_train_task():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    # 获取表单参数
    dataset_id = int(request.form['dataset_id'])
    algorithm_name = request.form['algorithm_name']
    n_clusters = request.form.get('n_clusters', None)
    
    # 处理聚类数量参数
    if n_clusters and n_clusters.strip():
        n_clusters = int(n_clusters)
    else:
        n_clusters = None
    
    try:
        run_training(dataset_id, algorithm_name, n_clusters)
        add_admin_log(f"开始训练模型，算法: {algorithm_name}，数据集ID: {dataset_id}")
        return redirect(url_for('admin.result_manage'))
    except Exception as e:
        # 训练失败，返回训练页并携带错误信息
        algorithms = ['kmeans', 'bgmm', 'fcm', 'meanshift']
        datasets = get_all_datasets()
        return render_template('admin/train.html', active_page='train', algorithms=algorithms, datasets=datasets, error=str(e))
# 5. 聚类结果页面
@admin_bp.route('/result')
def result_manage():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    # 查询所有模型+评估指标
    models_query = db.session.query(ModelInfo, EvaluationMetric, Dataset).join(
        EvaluationMetric, ModelInfo.model_id == EvaluationMetric.model_id
    ).join(Dataset, ModelInfo.dataset_id == Dataset.dataset_id).all()
    
    # 兼容数据库不存在remark字段的情况
    models = []
    for model, metric, dataset in models_query:
        if not hasattr(model, 'remark'):
            model.remark = ''
        models.append((model, metric, dataset))
    
    return render_template('admin/result.html', active_page='result', models=models)

# 6. 部署模型
# 切换模型部署状态
@admin_bp.route('/result/deploy/<int:model_id>')
def deploy_model(model_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    model = ModelInfo.query.get(model_id)
    if not model:
        return redirect(url_for('admin.result_manage'))
    # 切换部署状态
    model.is_deployed = not model.is_deployed
    db.session.commit()
    status = "部署" if model.is_deployed else "取消部署"
    add_admin_log(f"{status}模型ID: {model_id}")
    return redirect(url_for('admin.result_manage'))

# 7. 训练历史页面
@admin_bp.route('/history')
def train_history():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    # 查询所有训练任务
    tasks = db.session.query(TrainingTask, Dataset).join(Dataset, TrainingTask.dataset_id == Dataset.dataset_id).order_by(TrainingTask.create_time.desc()).all()
    return render_template('admin/history.html', active_page='history', tasks=tasks)

# 8. 用户管理页面
@admin_bp.route('/users', methods=['GET', 'POST'])
def user_manage():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    users = get_all_users()
    roles = UserRole.query.all()
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role_id = int(request.form['role_id'])
        
        # 验证管理员邀请码
        role = UserRole.query.get(role_id)
        if role.role_name == 'admin':
            invite_code = request.form.get('invite_code', '')
            if invite_code != Config.ADMIN_INVITE_CODE:
                return render_template('admin/user.html', active_page='user', users=users, roles=roles, error='管理员邀请码错误')
        
        success, msg = create_user(username, password, role_id)
        if success:
            add_admin_log(f"创建用户: {username}")
            return redirect(url_for('admin.user_manage'))
        else:
            return render_template('admin/user.html', active_page='user', users=users, roles=roles, error=msg)
    
    return render_template('admin/user.html', active_page='user', users=users, roles=roles)

# 9. 删除用户
@admin_bp.route('/users/delete/<int:user_id>')
def delete_user_route(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    delete_user(user_id)
    add_admin_log(f"删除用户ID: {user_id}")
    return redirect(url_for('admin.user_manage'))

# 10. 修改用户角色
@admin_bp.route('/users/update/<int:user_id>', methods=['POST'])
def update_user_role_route(user_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    new_role_id = int(request.form['new_role_id'])
    update_user_role(user_id, new_role_id)
    add_admin_log(f"修改用户ID: {user_id} 的角色为ID: {new_role_id}")
    return redirect(url_for('admin.user_manage'))

# 数据集详情/编辑页面
@admin_bp.route('/data/detail/<int:dataset_id>')
def data_detail(dataset_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    df, dataset, error = get_dataset_content(dataset_id)
    if error:
        return render_template('admin/data.html', active_page='data', datasets=get_all_datasets(), error=error)
    return render_template('admin/data_detail.html', active_page='data', dataset=dataset, df=df)

# 保存数据集修改
@admin_bp.route('/data/save/<int:dataset_id>', methods=['POST'])
def save_data_edit(dataset_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    new_data = request.get_json()
    success, msg = save_dataset_content(dataset_id, new_data)
    if success:
        add_admin_log(f"修改数据集ID: {dataset_id} 的内容")
    return {'success': success, 'msg': msg}

# 删除模型功能（修复已部署模型删除bug）
@admin_bp.route('/result/delete/<int:model_id>')
def delete_model(model_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    # 查询目标模型
    model = ModelInfo.query.get(model_id)
    if not model:
        return redirect(url_for('admin.result_manage'))
    
    try:
        # 1. 先自动取消模型部署（核心修复）
        model.is_deployed = False
        db.session.commit()
        
        # 2. 删除关联的评估指标数据（解决外键约束）
        EvaluationMetric.query.filter_by(model_id=model_id).delete()
        db.session.commit()
        
        # 3. 删除本地模型文件
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
        # 删除对应的标准化器文件
        if os.path.exists(model.scaler_path):
            os.remove(model.scaler_path)
        
        # 4. 删除该模型生成的所有可视化图片
        plot_dir = os.path.join('static', 'model_plots')
        plot_files = [
            f'model_{model_id}_scatter.png',
            f'model_{model_id}_accuracy.png',
            f'model_{model_id}_loss.png',
            f'model_{model_id}_loss_curve.png'
        ]
        for plot_file in plot_files:
            plot_path = os.path.join(plot_dir, plot_file)
            if os.path.exists(plot_path):
                os.remove(plot_path)
        
        # 5. 删除数据库中的模型记录
        db.session.delete(model)
        db.session.commit()
        add_admin_log(f"删除模型ID: {model_id}")
        
    except Exception as e:
        # 出错时回滚数据库，避免数据异常
        db.session.rollback()
    
    # 删除完成后刷新页面
    return redirect(url_for('admin.result_manage'))

# 保存模型备注
@admin_bp.route('/result/save-remark/<int:model_id>', methods=['POST'])
def save_model_remark(model_id):
    if session.get('role') != 'admin':
        return {'success': False, 'msg': '权限不足'}, 403
    
    model = ModelInfo.query.get(model_id)
    if not model:
        return {'success': False, 'msg': '模型不存在'}
    
    # 兼容数据库不存在remark字段的情况
    if not hasattr(model, 'remark'):
        return {'success': False, 'msg': '数据库版本不支持备注功能，请先执行数据库升级'}
    
    try:
        data = request.get_json()
        model.remark = data.get('remark', '')
        db.session.commit()
        return {'success': True, 'msg': '保存成功'}
    except Exception as e:
        db.session.rollback()
        return {'success': False, 'msg': str(e)}

# ==================== 备注管理路由 ====================
# 获取指定关联对象的备注列表
@admin_bp.route('/remark/list/<string:remark_type>/<int:relation_id>')
def get_admin_remark_list(remark_type, relation_id):
    if session.get('role') != 'admin':
        return {'success': False, 'msg': '权限不足'}, 403
    
    remarks = get_remark_list(remark_type, relation_id)
    # 格式化返回数据
    remark_list = []
    for remark in remarks:
        remark_list.append({
            'remark_id': remark.remark_id,
            'remark_type': remark.remark_type,
            'relation_id': remark.relation_id,
            'content': remark.remark_content,
            'is_active': remark.is_active,
            'creator_id': remark.creator_id,
            'create_time': remark.create_time.strftime('%Y-%m-%d %H:%M'),
            'update_time': remark.update_time.strftime('%Y-%m-%d %H:%M')
        })
    return {'success': True, 'data': remark_list}

# 创建备注
@admin_bp.route('/remark/create', methods=['POST'])
def create_admin_remark():
    if session.get('role') != 'admin':
        return {'success': False, 'msg': '权限不足'}, 403
    
    try:
        data = request.get_json()
        remark_type = data.get('remark_type')
        relation_id = data.get('relation_id')
        content = data.get('content')
        is_active = data.get('is_active', True)
        creator_id = session['user_id']
        
        success, msg = create_remark(remark_type, relation_id, content, is_active, creator_id)
        if success:
            return {'success': True, 'msg': msg}
        else:
            return {'success': False, 'msg': msg}, 400
    except Exception as e:
        return {'success': False, 'msg': f'请求异常：{str(e)}'}, 500

# 修改备注
@admin_bp.route('/remark/update/<int:remark_id>', methods=['POST'])
def update_admin_remark(remark_id):
    if session.get('role') != 'admin':
        return {'success': False, 'msg': '权限不足'}, 403
    
    try:
        data = request.get_json()
        content = data.get('content')
        is_active = data.get('is_active')
        
        success, msg = update_remark(remark_id, content, is_active)
        if success:
            return {'success': True, 'msg': msg}
        else:
            return {'success': False, 'msg': msg}, 400
    except Exception as e:
        return {'success': False, 'msg': f'请求异常：{str(e)}'}, 500

# 删除备注
@admin_bp.route('/remark/delete/<int:remark_id>', methods=['POST'])
def delete_admin_remark(remark_id):
    if session.get('role') != 'admin':
        return {'success': False, 'msg': '权限不足'}, 403
    
    success, msg = delete_remark(remark_id)
    if success:
        return {'success': True, 'msg': msg}
    else:
        return {'success': False, 'msg': msg}, 400

from flask import flash

# 备注管理页面（支持POST提交新增）
@admin_bp.route('/remark', methods=['GET', 'POST'])
def remark_manage():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        remark_content = request.form.get('remark_content')
        is_active = request.form.get('is_active') == 'on'
        creator_id = session['user_id']
        
        # 首页公告使用固定type=website，relation_id=1
        success, msg = create_remark('website', 1, remark_content, is_active, creator_id)
        if success:
            flash('公告发布成功', 'success')
        else:
            flash(f'发布失败：{msg}', 'error')
        return redirect(url_for('admin.remark_manage'))
    
    # 获取所有网站公告
    all_remarks = get_remark_list('website', 1)
    return render_template('admin/remark.html', active_page='remark', remarks=all_remarks)

# 编辑公告提交
@admin_bp.route('/remark/edit', methods=['POST'])
def edit_remark():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    remark_id = request.form.get('remark_id')
    remark_content = request.form.get('remark_content')
    is_active = request.form.get('is_active') == 'on'
    
    success, msg = update_remark(int(remark_id), remark_content, is_active)
    if success:
        flash('公告修改成功', 'success')
    else:
        flash(f'修改失败：{msg}', 'error')
    return redirect(url_for('admin.remark_manage'))

    # 删除公告
@admin_bp.route('/remark/delete/<int:remark_id>')
def delete_remark_route(remark_id):
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    success, msg = delete_remark(remark_id)
    if success:
        flash('公告删除成功', 'success')
        add_admin_log(f"删除公告ID: {remark_id}")
    else:
        flash(f'删除失败：{msg}', 'error')
    return redirect(url_for('admin.remark_manage'))

# 操作日志页面
@admin_bp.route('/logs')
def operation_logs():
    if session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    # 查询所有操作日志，关联用户表获取用户名
    logs = db.session.query(AdminOperationLog, User.username).join(
        User, AdminOperationLog.admin_id == User.user_id
    ).order_by(AdminOperationLog.create_time.desc()).all()
    
    return render_template('admin/logs.html', active_page='logs', logs=logs)
