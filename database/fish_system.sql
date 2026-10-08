-- 创建数据库
CREATE DATABASE IF NOT EXISTS fish_system CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE fish_system;

-- 1. 用户角色表
CREATE TABLE IF NOT EXISTS user_role (
    role_id INT AUTO_INCREMENT PRIMARY KEY,
    role_name VARCHAR(20) NOT NULL UNIQUE COMMENT 'admin/user'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 用户表
CREATE TABLE IF NOT EXISTS user (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL COMMENT '加密后的密码',
    role_id INT NOT NULL,
    FOREIGN KEY (role_id) REFERENCES user_role(role_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 数据集表
CREATE TABLE IF NOT EXISTS dataset (
    dataset_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    file_path VARCHAR(200) NOT NULL,
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    uploader_id INT NOT NULL,
    FOREIGN KEY (uploader_id) REFERENCES user(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 原始鱼类数据表
CREATE TABLE IF NOT EXISTS fish_data (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    species VARCHAR(50) COMMENT '真实标签',
    length FLOAT NOT NULL,
    weight FLOAT NOT NULL,
    l_w_ratio FLOAT NOT NULL,
    dataset_id INT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. 模型信息表
CREATE TABLE IF NOT EXISTS model_info (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL COMMENT 'kmeans/bgmm/fcm/meanshift',
    model_path VARCHAR(200) NOT NULL,
    scaler_path VARCHAR(200) NOT NULL,
    is_deployed BOOLEAN DEFAULT FALSE COMMENT '是否部署到前端',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    dataset_id INT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. 训练任务表
CREATE TABLE IF NOT EXISTS training_task (
    task_id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_id INT NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' COMMENT 'pending/running/completed',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dataset_id) REFERENCES dataset(dataset_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. 评估指标表
CREATE TABLE IF NOT EXISTS evaluation_metric (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    model_id INT NOT NULL,
    accuracy FLOAT COMMENT '准确率',
    loss_value FLOAT COMMENT '损失/替代指标',
    FOREIGN KEY (model_id) REFERENCES model_info(model_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8. 预测记录表
CREATE TABLE IF NOT EXISTS prediction_record (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    model_id INT NOT NULL COMMENT '使用的模型ID',
    model_name VARCHAR(100) NOT NULL COMMENT '模型名称',
    input_length FLOAT NOT NULL COMMENT '输入的长度(cm)',
    input_weight FLOAT NOT NULL COMMENT '输入的重量(g)',
    input_l_w_ratio FLOAT NOT NULL COMMENT '输入的长重比',
    predicted_label INT NOT NULL COMMENT '预测的聚类标签',
    predicted_species VARCHAR(50) NOT NULL COMMENT '预测的鱼类类别',
    confidence FLOAT COMMENT '预测置信度(%)',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '预测时间',
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES model_info(model_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 9. 预测结果表（已废弃，预测结果直接存放在prediction_record表中）
-- CREATE TABLE IF NOT EXISTS result (
--     result_id INT AUTO_INCREMENT PRIMARY KEY,
--     prediction_record_id INT NOT NULL,
--     predicted_species VARCHAR(50) NOT NULL,
--     confidence FLOAT,
--     FOREIGN KEY (prediction_record_id) REFERENCES prediction_record(record_id) ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 10. 管理员操作日志表
CREATE TABLE IF NOT EXISTS admin_operation_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    admin_id INT NOT NULL,
    operation VARCHAR(100) NOT NULL,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES user(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 11. 数据备注信息表
CREATE TABLE IF NOT EXISTS data_remark (
    remark_id INT AUTO_INCREMENT PRIMARY KEY,
    remark_type VARCHAR(20) NOT NULL COMMENT '备注关联类型：dataset/模型model',
    relation_id INT NOT NULL COMMENT '关联主体ID，对应dataset_id/model_id',
    remark_content TEXT NOT NULL COMMENT '备注详情内容',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否对普通用户可见',
    creator_id INT NOT NULL COMMENT '创建人ID（管理员）',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    FOREIGN KEY (creator_id) REFERENCES user(user_id) ON DELETE CASCADE,
    INDEX idx_relation_query (remark_type, relation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理员维护的备注信息表，普通用户仅可查看启用状态的内容';

-- 插入初始角色数据
INSERT IGNORE INTO user_role (role_name) VALUES ('admin'), ('user');