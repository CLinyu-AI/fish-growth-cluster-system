// 验证码刷新
function refreshCaptcha() {
    const captchaImg = document.getElementById('captcha-img');
    captchaImg.src = captchaImg.src.split('?')[0] + '?t=' + Math.random();
}

// 通用确认弹窗
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 绑定验证码刷新事件
    const captchaImg = document.getElementById('captcha-img');
    if (captchaImg) {
        captchaImg.addEventListener('click', refreshCaptcha);
    }

    // 绑定所有删除按钮的确认弹窗
    const deleteButtons = document.querySelectorAll('.btn-danger');
    deleteButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            if (!confirm('确定要执行删除操作吗？此操作不可恢复！')) {
                e.preventDefault();
            }
        });
    });
});