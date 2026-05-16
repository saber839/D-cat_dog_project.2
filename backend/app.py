from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2
from model_handler import ModelHandler

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载模型
MODEL_PATH = '../models/best_cnn_model.pth'
model_handler = ModelHandler(MODEL_PATH)

print("模型加载完成，服务已启动！")


def image_to_base64(image_np):
    """将numpy图片转为base64字符串"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': '猫狗分类服务运行中'})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    预测接口
    接收：multipart/form-data，字段名 'image'
    返回：预测结果JSON
    """
    if 'image' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    try:
        # 读取图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # 预测
        result = model_handler.predict(image)

        # 生成Grad-CAM
        gradcam_image = model_handler.generate_gradcam(image)
        gradcam_base64 = image_to_base64(gradcam_image)

        # 原图转base64（用于前端显示）
        original_np = np.array(image.resize((224, 224)))
        original_base64 = image_to_base64(original_np)

        result['original_image'] = f'data:image/jpeg;base64,{original_base64}'
        result['gradcam_image'] = f'data:image/jpeg;base64,{gradcam_base64}'

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'处理图片时出错: {str(e)}'}), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量预测接口
    接收：multipart/form-data，字段名 'images'（可多个文件）
    返回：预测结果列表
    """
    if 'images' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400

    files = request.files.getlist('images')
    results = []

    for file in files:
        if file.filename == '':
            continue

        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            result = model_handler.predict(image)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results, 'total': len(results)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)