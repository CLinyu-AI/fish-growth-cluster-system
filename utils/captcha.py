import random
import string
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from flask import session

def generate_captcha_code(length=4):
    """生成随机数字验证码"""
    return ''.join(random.choices(string.digits, k=length))

def generate_captcha_image(code):
    """生成验证码图片"""
    width, height = 120, 40
    image = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    
    # 尝试加载系统字体，失败则用默认
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 画验证码
    for i, char in enumerate(code):
        x = 20 + i * 25
        y = random.randint(5, 10)
        draw.text((x, y), char, font=font, fill=(0, 0, 0))
    
    # 加干扰线
    for _ in range(3):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(180, 180, 180), width=1)
    
    # 保存到内存
    buf = BytesIO()
    image.save(buf, 'png')
    buf.seek(0)
    return buf

def verify_captcha(input_code):
    """验证验证码"""
    if 'captcha' not in session:
        return False
    return input_code.lower() == session['captcha'].lower()