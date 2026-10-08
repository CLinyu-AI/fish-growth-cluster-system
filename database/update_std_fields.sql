-- 为evaluation_metric表新增标准差字段
ALTER TABLE evaluation_metric 
ADD COLUMN accuracy_std DECIMAL(10,4) DEFAULT 0.0000 COMMENT '准确率标准差' AFTER ch_score,
ADD COLUMN silhouette_std DECIMAL(10,4) DEFAULT 0.0000 COMMENT '轮廓系数标准差' AFTER accuracy_std,
ADD COLUMN dbi_std DECIMAL(10,4) DEFAULT 0.0000 COMMENT 'DBI指数标准差' AFTER silhouette_std,
ADD COLUMN ch_std DECIMAL(10,4) DEFAULT 0.0000 COMMENT 'CH指数标准差' AFTER dbi_std;