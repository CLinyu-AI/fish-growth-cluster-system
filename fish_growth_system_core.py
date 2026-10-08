#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鱼类生长分析系统 核心代码提取
包含系统最关键的核心实现部分，非核心逻辑已做注释说明
"""


# ==============================================
# 1. 系统整体框架和主流程
# ==============================================
# 初始化Flask应用
app = Flask(__name__)
app.config.from_object('config.Config')
db = SQLAlchemy(app)

# 配置说明（完整配置见config.py）
"""
核心配置项：
- SQLALCHEMY_DATABASE_URI: MySQL数据库连接地址
- SECRET_KEY: Flask会话密钥
- UPLOAD_FOLDER: 数据集上传目录
- SAVED_MODELS_FOLDER: 训练模型保存目录
- ALLOWED_EXTENSIONS: 允许上传的文件格式
"""

# 系统主入口（完整代码见app.py）
if __name__ == '__main__':
    # 初始化数据库
    with app.app_context():
        db.create_all()
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=True)

# ==============================================
# 2. 特征工程与数据处理
# ==============================================
class DataPreprocessor:
    """数据预处理核心类，实现标准化、PCA降维等功能"""
    
    def __init__(self, use_pca=True, variance_threshold=0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold) if use_pca else None
        self.use_pca = use_pca
        self.feature_names = None
    
    def fit_transform(self, df, features=['length', 'weight', 'w_l_ratio']):
        """训练预处理模型并转换数据"""
        self.feature_names = features
        # 标准化处理
        X_scaled = self.scaler.fit_transform(df[features])
        
        # PCA降维处理（解决特征多重共线性问题）
        if self.use_pca:
            X_processed = self.pca.fit_transform(X_scaled)
            print(f"PCA降维后保留维度: {X_processed.shape[1]}, 保留方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            X_processed = X_scaled
            
        return X_processed
    
    def transform(self, df):
        """对新数据进行转换"""
        X_scaled = self.scaler.transform(df[self.feature_names])
        if self.use_pca:
            return self.pca.transform(X_scaled)
        return X_scaled
    
    def save(self, save_path):
        """保存预处理模型"""
        joblib.dump(self, save_path)
    
    @staticmethod
    def load(load_path):
        """加载预处理模型"""
        return joblib.load(load_path)

# 数据校验和加载（非核心逻辑已简化）
def load_dataset(file_path):
    """加载并校验CSV数据集"""
    df = pd.read_csv(file_path)
    required_cols = ['length', 'weight']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("数据集缺少必要字段: length, weight")
    # 自动计算重长比特征
    df['w_l_ratio'] = df['weight'] / df['length']
    return df

# ==============================================
# 3. 模型训练与评估
# ==============================================
# 3.1 聚类评估指标计算
def calculate_clustering_metrics(X, labels):
    """计算核心聚类评估指标"""
    metrics = {}
    if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
        metrics['silhouette_score'] = silhouette_score(X, labels)
        metrics['ch_score'] = calinski_harabasz_score(X, labels)
        metrics['db_score'] = davies_bouldin_score(X, labels)
    else:
        metrics['silhouette_score'] = 0
        metrics['ch_score'] = 0
        metrics['db_score'] = float('inf')
    return metrics

def calculate_ari(y_true, y_pred):
    """计算调整兰德指数（ARI），衡量聚类结果与真实标签的匹配度"""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred) if len(y_true) == len(y_pred) else 0

# 3.2 自动超参数调优核心（简化版，完整实现见utils/auto_tune.py）
def auto_select_best_params(X, algorithm='kmeans'):
    """自动选择最优超参数"""
    best_params = {}
    best_score = -float('inf')
    
    if algorithm.lower() == 'kmeans':
        # KMeans自动搜索最佳聚类数
        max_clusters = min(20, len(X)//10)
        for n_clusters in range(2, max_clusters+1):
            model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels) if len(np.unique(labels))>1 else -1
            if score > best_score:
                best_score = score
                best_params['n_clusters'] = n_clusters
                
    elif algorithm.lower() == 'bgmm':
        # BGMM高斯混合模型调优
        max_components = min(20, len(X)//10)
        for n_components in range(2, max_components+1):
            model = GaussianMixture(n_components=n_components, n_init=10, random_state=42)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels) if len(np.unique(labels))>1 else -1
            if score > best_score:
                best_score = score
                best_params['n_components'] = n_components
                
    # 其他算法（FCM、MeanShift）调优逻辑类似，完整实现见auto_tune.py
    return best_params, best_score

# 3.3 模型训练主流程
def train_clustering_model(X, algorithm='kmeans', params=None, auto_tune=True):
    """训练聚类模型主函数"""
    # 自动调优参数
    if auto_tune:
        params, _ = auto_select_best_params(X, algorithm)
    
    # 初始化模型
    model = None
    if algorithm.lower() == 'kmeans':
        model = KMeans(**params, n_init=20, random_state=42)
    elif algorithm.lower() == 'bgmm':
        model = GaussianMixture(**params, n_init=10, random_state=42)
    elif algorithm.lower() == 'fcm':
        # FCM模糊聚类实现（完整代码见algorithms/fcm.py）
        class FCMWrapper:
            def __init__(self, n_clusters, m=2, max_iter=2000):
                self.n_clusters = n_clusters
                self.m = m
                self.max_iter = max_iter
                self.centers = None
                
            def fit_predict(self, X):
                cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    X.T, self.n_clusters, self.m, error=1e-7, maxiter=self.max_iter, seed=42
                )
                self.centers = cntr
                return np.argmax(u, axis=0)
        model = FCMWrapper(**params)
    elif algorithm.lower() == 'meanshift':
        bandwidth = estimate_bandwidth(X, quantile=0.2)
        model = MeanShift(bandwidth=bandwidth, bin_seeding=False)
    
    # 训练并返回结果
    labels = model.fit_predict(X)
    metrics = calculate_clustering_metrics(X, labels)
    return model, labels, metrics

# ==============================================
# 4. 模型解释与可视化
# ==============================================
def generate_cluster_scatter_plot(X, labels, save_path=None):
    """生成聚类散点图（基于ECharts的前端实现核心逻辑）
    完整可视化实现见utils/plot_utils.py和前端模板
    """
    # 降维到2D用于可视化
    pca_vis = PCA(n_components=2)
    X_2d = pca_vis.fit_transform(X)
    
    # 按聚类分组
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X_2d[i].tolist())
    
    # 返回ECharts所需格式
    return {
        'clusters': clusters,
        'explained_variance': pca_vis.explained_variance_ratio_.tolist()
    }

# 可视化图表生成逻辑简化说明
"""
其他可视化图表实现：
1. 特征分布直方图：使用seaborn绘制各特征的分布情况
2. 混淆矩阵：当有真实标签时生成聚类结果与真实标签的对比矩阵
3. 指标雷达图：对比不同算法的各项评估指标
4. 学习曲线：展示不同参数下的评估指标变化趋势
所有图表均支持PNG导出和前端ECharts交互式展示
"""

# ==============================================
# 5. Flask后端核心代码
# ==============================================
# 5.1 用户身份验证装饰器（简化版）
def login_required(f):
    """用户登录验证装饰器"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'code': 401, 'msg': '请先登录'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# 5.2 批量预测接口核心实现
@app.route('/user/batch_predict', methods=['GET', 'POST'])
@login_required
def batch_predict():
    """批量预测接口"""
    if request.method == 'POST':
        # 1. 获取上传文件和模型ID
        file = request.files['csv_file']
        model_id = request.form.get('model_id')
        
        # 2. 加载模型和预处理对象
        model_info = ModelInfo.query.get(model_id)
        model = joblib.load(model_info.model_path)
        preprocessor = DataPreprocessor.load(model_info.preprocessor_path)
        
        # 3. 加载并处理数据
        df = load_dataset(file)
        X_processed = preprocessor.transform(df)
        
        # 4. 执行预测
        labels = model.predict(X_processed)
        # 计算置信度（根据距离聚类中心的远近等逻辑）
        confidence = calculate_confidence(model, X_processed, labels)
        
        # 5. 组装结果
        results = []
        for i in range(len(df)):
            results.append({
                'index': i+1,
                'length': round(float(df.iloc[i]['length']), 2),
                'weight': round(float(df.iloc[i]['weight']), 2),
                'w_l_ratio': round(float(df.iloc[i]['w_l_ratio']), 4),
                'predicted_label': int(labels[i]),
                'predicted_species': get_species_name(labels[i], model_info.dataset_id),
                'confidence': round(float(confidence[i]) * 100, 1)
            })
        
        # 6. 保存预测记录到数据库
        save_prediction_records(session['user_id'], model_id, results)
        
        # 7. 返回结果页面
        return render_template('user/batch_predict.html', results=results, models=get_deployed_models())
    
    # GET请求返回页面
    return render_template('user/batch_predict.html', models=get_deployed_models())

# 辅助函数（非核心实现已注释）
"""
其他核心接口实现：
- 用户登录/注册：用户身份认证、权限管理
- 数据集上传/管理：用户数据集的上传、预览、删除
- 模型训练接口：异步处理训练任务，实时返回训练进度
- 结果查询接口：查询训练历史和预测记录
- 管理员接口：用户管理、系统配置、日志查看
完整路由实现见routes/目录下的各路由文件
"""

# ==============================================
# 数据模型定义（简化版，完整实现见models/目录）
# ==============================================
class User(db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # admin/user

class ModelInfo(db.Model):
    """模型信息模型"""
    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    model_path = db.Column(db.String(255), nullable=False)
    preprocessor_path = db.Column(db.String(255), nullable=False)
    dataset_id = db.Column(db.Integer, nullable=False)
    ari = db.Column(db.Float)
    silhouette_score = db.Column(db.Float)
    is_deployed = db.Column(db.Boolean, default=False)
    create_time = db.Column(db.DateTime, default=datetime.now)

# 其他模型定义（数据集、训练任务、评估指标等）见models/目录

# ==============================================
# 扩展功能说明
# ==============================================
"""
其他系统核心功能（未完全展开）：
1. 增广数据集集成：自动加载多份增广数据集进行模型二次微调，提升泛化能力
2. 模型稳定性评估：在多份数据集上评估模型，计算指标标准差量化稳定性
3. 消融实验自动记录：自动保存不同参数下的实验结果，便于对比分析
4. 操作日志系统：记录所有管理员操作和用户关键行为
5. 权限控制：细粒度的用户权限管理，区分普通用户和管理员功能
6. 异步任务处理：训练任务异步执行，避免阻塞前端请求
"""