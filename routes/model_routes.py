from flask import Blueprint, request, render_template, jsonify
# 核心修正：从services包导入predict_service的函数
from services.predict_service import get_deployed_models, predict_fish
import traceback

# 创建蓝图（需在主应用中注册该蓝图）
model_bp = Blueprint('model', __name__, url_prefix='/model')

@model_bp.route('/predict', methods=['GET', 'POST'])
def fish_predict():
    """鱼类生长特征聚类预测页面路由（GET：展示表单；POST：执行预测）"""
    error = None
    result = None
    
    # 1. GET请求：加载已部署模型列表，渲染预测表单
    if request.method == 'GET':
        try:
            deployed_models = get_deployed_models()
            return render_template(
                'user/predict.html',  # 修正：匹配你的项目模板路径
                models=deployed_models,
                error=error,
                result=result
            )
        except Exception as e:
            error = f"加载模型列表失败：{str(e)}"
            return render_template('user/predict.html', models=[], error=error, result=result)
    
    # 2. POST请求：处理预测提交
    elif request.method == 'POST':
        try:
            # 提取并验证表单参数
            model_id = request.form.get('model_id')
            length = float(request.form.get('length'))
            weight = float(request.form.get('weight'))
            w_l_ratio = float(request.form.get('w_l_ratio'))
            
            # 参数合法性校验
            if length <= 0 or weight <= 0 or w_l_ratio <= 0:
                error = "输入参数错误：长度/重量/重长比必须为正数"
                deployed_models = get_deployed_models()
                return render_template('user/predict.html', models=deployed_models, error=error, result=result)
            
            # 调用预测服务
            predict_result, predict_error = predict_fish(model_id, length, weight, w_l_ratio)
            if predict_error:
                error = predict_error
            else:
                result = predict_result
            
            # 重新加载模型列表（保持下拉框数据）
            deployed_models = get_deployed_models()
            return render_template(
                'user/predict.html',
                models=deployed_models,
                error=error,
                result=result
            )
        
        except ValueError as e:
            error = f"参数格式错误：{str(e)}（请输入有效的数字）"
            deployed_models = get_deployed_models()
            return render_template('user/predict.html', models=deployed_models, error=error, result=result)
        
        except Exception as e:
            error = f"预测处理异常：{str(e)}"
            traceback.print_exc()
            deployed_models = get_deployed_models()
            return render_template('user/predict.html', models=deployed_models, error=error, result=result)

# 可选：JSON格式预测API（适配AJAX/第三方调用）
@model_bp.route('/api/predict', methods=['POST'])
def api_fish_predict():
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        length = float(data.get('length'))
        weight = float(data.get('weight'))
        w_l_ratio = float(data.get('w_l_ratio'))
        
        if not all([model_id, length, weight, w_l_ratio]):
            return jsonify({
                'code': 400,
                'msg': '参数缺失：必须提供model_id、length、weight、w_l_ratio',
                'data': None
            })
        if length <= 0 or weight <= 0 or w_l_ratio <= 0:
            return jsonify({
                'code': 400,
                'msg': '参数错误：长度/重量/重长比必须为正数',
                'data': None
            })
        
        predict_result, predict_error = predict_fish(model_id, length, weight, w_l_ratio)
        if predict_error:
            return jsonify({
                'code': 500,
                'msg': predict_error,
                'data': None
            })
        
        return jsonify({
            'code': 200,
            'msg': '预测成功',
            'data': predict_result
        })
    
    except ValueError as e:
        return jsonify({
            'code': 400,
            'msg': f'参数格式错误：{str(e)}',
            'data': None
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'code': 500,
            'msg': f'服务器内部错误：{str(e)}',
            'data': None
        })

# 可选：获取已部署模型列表API
@model_bp.route('/api/models', methods=['GET'])
def api_get_models():
    try:
        models = get_deployed_models()
        model_list = [{
            'model_id': model.model_id,
            'model_name': model.model_name,
            'accuracy': model.accuracy,
            'remark': model.remark,
            'dataset_id': model.dataset_id
        } for model in models]
        return jsonify({
            'code': 200,
            'msg': '获取模型列表成功',
            'data': model_list
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': f'获取模型列表失败：{str(e)}',
            'data': None
        })