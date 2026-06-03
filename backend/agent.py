# -*- coding: utf-8 -*-
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os

import numpy as np
import requests


class CatDogAgent:

    def __init__(self, model_handler, llm_api_key=None, llm_base_url=None):
        self.model_handler = model_handler
        self.llm_api_key = llm_api_key or os.environ.get('LLM_API_KEY', '')
        self.llm_base_url = llm_base_url or os.environ.get(
            'LLM_BASE_URL',
            'https://api.openai.com/v1/chat/completions'
        )
        self.conversation_history = []
        self.last_analysis = None

    def predict_image(self, image):
        return self.model_handler.predict(image)

    def generate_heatmap(self, image):
        return self.model_handler.generate_gradcam(image)

    def call_llm(self, user_prompt=None, system_prompt=None, messages=None):
        if not self.llm_api_key:
            return None

        msg_list = []
        if system_prompt:
            msg_list.append({'role': 'system', 'content': system_prompt})
        if messages:
            msg_list.extend(messages)
        elif user_prompt:
            msg_list.append({'role': 'user', 'content': user_prompt})
        else:
            return None

        try:
            payload = {
                'model': 'deepseek-chat',
                'messages': msg_list,
                'temperature': 0.7,
                'max_tokens': 500,
            }

            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

            response = requests.post(
                self.llm_base_url,
                headers={
                    'Authorization': f'Bearer {self.llm_api_key}',
                    'Content-Type': 'application/json; charset=utf-8',
                },
                data=body,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print('[Agent] LLM returned content, length:', len(content))
                return content

            print('[Agent] LLM status not 200:', response.status_code)
            print('[Agent] Response body:', response.text[:200])
            return None

        except Exception as e:
            print('[Agent] Exception:', type(e).__name__, str(e)[:100])
            return None

    def _analyze_focus(self, gradcam_image):
        red_channel = gradcam_image[:, :, 0]
        high_activation = (red_channel > 200).sum()
        total_pixels = red_channel.size
        focus_ratio = high_activation / total_pixels

        if focus_ratio > 0.3:
            desc = '模型关注区域较为分散，可能受背景信息影响'
        elif focus_ratio > 0.1:
            desc = '模型关注区域集中，聚焦于动物主体特征'
        else:
            desc = '模型仅关注极少数关键点，预测基于核心特征'

        return {'ratio': round(focus_ratio, 3), 'description': desc}

    def analyze_image(self, image, user_question=None):
        self.conversation_history = []

        prediction = self.predict_image(image)
        gradcam = self.generate_heatmap(image)
        focus_analysis = self._analyze_focus(gradcam)

        self.last_analysis = {
            'prediction': prediction,
            'focus_analysis': focus_analysis,
        }

        if self.llm_api_key:
            print('[Agent] KEY EXISTS, calling LLM...')
            llm_result = self._generate_llm_response(
                prediction, focus_analysis, user_question
            )
            print('[Agent] LLM result type:', type(llm_result).__name__)
            if llm_result and len(str(llm_result)) > 10:
                print('[Agent] Using LLM response')
                agent_response = llm_result
            else:
                print('[Agent] LLM returned empty/short, fallback to rule')
                agent_response = self._generate_rule_response(
                    prediction, focus_analysis, user_question
                )
        else:
            print('[Agent] NO KEY, using rule mode')
            agent_response = self._generate_rule_response(
                prediction, focus_analysis, user_question
            )

        self.conversation_history.append({
            'role': 'user',
            'content': user_question or '请分析这张图片',
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': agent_response,
        })

        return {
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'agent_response': agent_response,
            'focus_analysis': focus_analysis,
            'gradcam_image': gradcam,
        }

    def _generate_llm_response(self, prediction, focus_analysis, user_question):
        system_prompt = '你是一个专业的计算机视觉分析助手。回复专业但易懂，控制在200字以内。'

        user_prompt = f"""
图片预测结果：
- 类别：{prediction['prediction']}
- 置信度：{prediction['confidence']:.2f}%
- 各分类概率：{prediction['probabilities']}
- Grad-CAM分析：{focus_analysis['description']}
- 用户问题：{user_question or '请给出分析报告'}
"""
        return self.call_llm(user_prompt=user_prompt, system_prompt=system_prompt)

    def _generate_rule_response(self, prediction, focus_analysis, user_question):
        if user_question:
            if '为什么' in user_question:
                return f"模型将该图预测为{prediction['prediction']}，置信度{prediction['confidence']:.1f}%。{focus_analysis['description']}。"
            if '准确' in user_question or '可信' in user_question:
                level = '高' if prediction['confidence'] > 85 else '中'
                return f'本次预测可信度为{level}。模型在4000张校园场景数据上训练，验证准确率92%。'
            return f'关于「{user_question}」，请提供更具体的问题。'

        return self._generate_template_report()

    def _generate_template_report(self):
        if not self.last_analysis:
            return '请先上传图片进行分析。'

        result = self.last_analysis['prediction']
        focus = self.last_analysis['focus_analysis']

        confidence_level = '✅ 置信度较高' if result['confidence'] > 85 else '⚠️ 置信度中等'

        return f"""
📊 猫狗识别分析报告
═══════════════════════
🔮 预测: {result['prediction']}
📈 置信度: {result['confidence']:.2f}%
🔍 模型关注: {focus['description']}
💡 {confidence_level}
"""

    def _build_chat_system_prompt(self):
        prediction = self.last_analysis['prediction']
        focus = self.last_analysis['focus_analysis']
        return (
            '你是猫狗识别系统的AI助手。基于以下分析上下文回答用户问题：\n'
            f"- 预测：{prediction['prediction']}\n"
            f"- 置信度：{prediction['confidence']:.2f}%\n"
            f"- Grad-CAM：{focus['description']}"
        )

    def chat(self, user_message):
        if not self.last_analysis:
            return '请先上传一张图片。'

        prediction = self.last_analysis['prediction']
        focus = self.last_analysis['focus_analysis']

        self.conversation_history.append({'role': 'user', 'content': user_message})

        if self.llm_api_key:
            result = self.call_llm(
                messages=self.conversation_history.copy(),
                system_prompt=self._build_chat_system_prompt(),
            )
            if result:
                self.conversation_history.append({'role': 'assistant', 'content': result})
                return result

        response = self._generate_rule_response(prediction, focus, user_message)
        self.conversation_history.append({'role': 'assistant', 'content': response})
        return response

    def clear_history(self):
        self.conversation_history = []
        self.last_analysis = None
