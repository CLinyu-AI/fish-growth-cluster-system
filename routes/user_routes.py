from flask import Blueprint, render_template, request, session, redirect, url_for, make_response
from services.predict_service import get_deployed_models, predict_fish, batch_predict_fish
import pandas as pd
import io
import csv
from services.data_service import get_remark_list
from models import PredictionRecord
from utils.database import db

user_bp = Blueprint('user', __name__)

# 用户首页
@user_bp.route('/')
def dashboard():
    if session.get('role') != 'user':
        return redirect(url_for('auth.login'))
    # 获取首页展示的备注信息（与后台管理对应）
    remarks = get_remark_list('website', 1, only_active=True)
    return render_template('user/dashboard.html', active_page='dashboard', remarks=remarks)

# 预测页面
@user_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if session.get('role') != 'user':
        return redirect(url_for('auth.login'))
    # 获取已部署的模型
    models = get_deployed_models()
    result = None
    error = None
    
    if request.method == 'POST':
        model_id = int(request.form['model_id'])
        length = float(request.form['length'])
        weight = float(request.form['weight'])
        w_l_ratio = float(request.form['w_l_ratio'])
        
        result, error = predict_fish(model_id, length, weight, w_l_ratio)
        
        # 预测成功则保存记录
        if result and not error:
            model = get_deployed_models()
            model_name = next((m.model_name for m in model if m.model_id == model_id), '未知模型')
            new_record = PredictionRecord(
                user_id=session['user_id'],
                model_id=model_id,
                model_name=model_name,
                input_length=length,
                input_weight=weight,
                input_l_w_ratio=w_l_ratio,
                predicted_label=result['predicted_label'],
                predicted_species=result['predicted_species'],
                confidence=result['confidence']
            )
            db.session.add(new_record)
            db.session.commit()
    
    return render_template('user/predict.html', active_page='predict', models=models, result=result, error=error)


# 预测记录页面
@user_bp.route('/records')
def predict_records():
    if session.get('role') != 'user':
        return redirect(url_for('auth.login'))
    
    # 获取当前用户的最新500条预测记录，按时间倒序，优化加载速度
    records = PredictionRecord.query.filter_by(user_id=session['user_id']).order_by(PredictionRecord.create_time.desc()).limit(500).all()
    return render_template('user/records.html', active_page='records', records=records)

# 普通用户获取可见的备注列表
@user_bp.route('/remark/list/<string:remark_type>/<int:relation_id>')
def get_user_remark_list(remark_type, relation_id):
    if not session.get('user_id'):
        return {'success': False, 'msg': '请先登录'}, 401
    
    # 只查询启用状态的备注
    remarks = get_remark_list(remark_type, relation_id, only_active=True)
    remark_list = []
    for remark in remarks:
        remark_list.append({
            'content': remark.remark_content,
            'update_time': remark.update_time.strftime('%Y-%m-%d %H:%M')
        })
    return {'success': True, 'data': remark_list}

# 批量预测页面
@user_bp.route('/batch-predict', methods=['GET', 'POST'])
def batch_predict():
    if session.get('role') != 'user':
        return redirect(url_for('auth.login'))
    
    models = get_deployed_models()
    results = None
    error = None
    download_url = None
    
    if request.method == 'POST':
        model_id = int(request.form['model_id'])
        file = request.files.get('csv_file')
        
        if not file or file.filename == '':
            error = "请选择CSV文件"
        elif not file.filename.endswith('.csv'):
            error = "请上传CSV格式的文件"
        else:
            try:
                # 读取CSV文件
                df = pd.read_csv(file)
                # 验证列是否存在
                required_columns = ['length', 'weight', 'w_l_ratio']
                if not all(col in df.columns for col in required_columns):
                    error = "CSV文件必须包含length、weight、w_l_ratio三列"
                else:
                    # 执行批量预测
                    results, error = batch_predict_fish(model_id, df)
                    if results and not error:
                        # 生成结果CSV供下载
                        output = io.StringIO()
                        writer = csv.writer(output)
                        # 写入表头
                        writer.writerow(['序号', '长度(cm)', '重量(g)', '重长比', '聚类标签', '预测物种', '置信度(%)'])
                        # 写入数据
                        for res in results:
                            writer.writerow([
                                res['index'],
                                res['length'],
                                res['weight'],
                                res['w_l_ratio'],
                                res['predicted_label'],
                                res['predicted_species'],
                                res['confidence']
                            ])
                        # 保存到session供下载
                        output.seek(0)
                        session['batch_predict_results'] = output.getvalue()
                        download_url = url_for('user.download_batch_results')
            except Exception as e:
                error = f"文件处理失败：{str(e)}"
    
    return render_template('user/batch_predict.html', active_page='batch_predict', models=models, results=results, error=error, download_url=download_url)

# 下载批量预测结果
@user_bp.route('/download-batch-results')
def download_batch_results():
    if session.get('role') != 'user' or 'batch_predict_results' not in session:
        return redirect(url_for('user.batch_predict'))
    
    response = make_response(session['batch_predict_results'])
    response.headers["Content-Disposition"] = f"attachment; filename=predict_results.csv"
    response.headers["Content-type"] = "text/csv"
    return response
