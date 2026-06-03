import os
import io
import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from model_handler import ModelHandler
from agent import CatDogAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_cnn_model.pth')

MODEL_PATH = os.environ.get('MODEL_PATH', DEFAULT_MODEL_PATH)
LLM_API_KEY = os.environ.get('LLM_API_KEY', '').strip()
LLM_BASE_URL = os.environ.get(
    'LLM_BASE_URL',
    'https://api.deepseek.com/v1/chat/completions'
)

app = Flask(__name__)
CORS(app)

model_handler = ModelHandler(MODEL_PATH)
agent = CatDogAgent(
    model_handler=model_handler,
    llm_api_key=LLM_API_KEY or None,
    llm_base_url=LLM_BASE_URL,
)


def image_to_base64(image_np):
    """将numpy图片转为base64字符串"""
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'message': '猫狗分类服务运行中',
        'llm_enabled': bool(LLM_API_KEY),
        'model_loaded': True,
        'model_path': MODEL_PATH,
    })


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
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        result = model_handler.predict(image)

        gradcam_image = model_handler.generate_gradcam(image)
        gradcam_base64 = image_to_base64(gradcam_image)

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

            gradcam_image = model_handler.generate_gradcam(image)
            original_np = np.array(image.resize((224, 224)))
            result['original_image'] = f'data:image/jpeg;base64,{image_to_base64(original_np)}'
            result['gradcam_image'] = f'data:image/jpeg;base64,{image_to_base64(gradcam_image)}'

            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify({'results': results, 'total': len(results)})


@app.route('/api/agent/analyze', methods=['POST'])
def agent_analyze():
    """Agent分析接口"""
    if 'image' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400

    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'无法读取图片: {str(e)}'}), 400

    result = agent.analyze_image(image)

    original_np = np.array(image.resize((224, 224)))
    result['original_image'] = f'data:image/jpeg;base64,{image_to_base64(original_np)}'
    gradcam_b64 = image_to_base64(result['gradcam_image'])
    result['gradcam_image'] = f'data:image/jpeg;base64,{gradcam_b64}'

    return jsonify(result)


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """Agent对话接口"""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': '消息为空'}), 400

    response = agent.chat(user_message)
    return jsonify({'response': response})


@app.route('/api/agent/reset', methods=['POST'])
def agent_reset():
    """重置Agent对话"""
    agent.clear_history()
    return jsonify({'status': 'ok', 'message': '对话已重置'})


@app.route('/')
def serve_index():
    """提供前端首页"""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_frontend(filename):
    """提供前端静态资源（CSS/JS/示例图片等）"""
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == '__main__':
    mode = 'LLM增强' if LLM_API_KEY else '规则模式'
    print(f'服务启动 | 模式: {mode} | 模型: {MODEL_PATH}')
    print('请在浏览器打开: http://127.0.0.1:5000  （不要直接双击 index.html）')
    app.run(host='0.0.0.0', port=5000, debug=True)
