from models import AdminOperationLog, db
from flask import session

def add_admin_log(operation):
    """
    记录管理员操作日志
    :param operation: 操作描述字符串
    """
    try:
        admin_id = session.get('user_id')
        if admin_id and session.get('role') == 'admin':
            log = AdminOperationLog(
                admin_id=admin_id,
                operation=operation
            )
            db.session.add(log)
            db.session.commit()
    except Exception as e:
        print(f"记录日志失败: {str(e)}")
        db.session.rollback()