from flask import Flask, render_template, request, jsonify, Response
import requests
import json

app = Flask(__name__)

# 后端服务配置
BACKEND_URL = "http://localhost:8000"

# 模型状态
def get_model_status():
    """获取模型状态"""
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        return response.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}

# 发送推理请求
def send_completion_request(prompt):
    """发送推理请求"""
    try:
        # 直接使用前端构建的完整提示词，不再重复添加前缀
        
        # 记录发送给模型的提示词
        print(f"后端发送给模型的提示词: {prompt}")
        
        # 发送请求，设置stream: true
        response = requests.post(
            f"{BACKEND_URL}/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "rwkv7-g1b-1.5b-20251202-ctx8192",
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.3,
                "stream": True
            }),
            stream=True
        )
        return response
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """主页面"""
    status = get_model_status()
    return render_template('index.html', status=status)

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天API - 支持流式响应"""
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # 获取流式响应
    response = send_completion_request(prompt)
    
    if isinstance(response, dict) and "error" in response:
        return jsonify(response), 500
    
    # 使用生成器返回SSE响应
    def generate():
        try:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    # 解码并处理每个SSE事件
                    chunk = chunk.decode('utf-8')
                    # 分割多个事件
                    events = chunk.split('\n\n')
                    for event in events:
                        if event.strip():
                            # 过滤掉data: [DONE]标记
                            if event.strip() == 'data: [DONE]':
                                continue
                            # 移除"data: "前缀
                            if event.startswith('data: '):
                                event = event[6:]
                            # 发送事件
                            yield f"event: message\ndata: {event}\n\n"
        finally:
            response.close()
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/status')
def status():
    """状态API"""
    status = get_model_status()
    return jsonify(status)

@app.route('/api/restart-model', methods=['POST'])
def restart_model():
    """重启模型API"""
    try:
        # 首先获取当前模型路径和状态
        current_status = get_model_status()
        
        # 使用相同的模型路径和策略重启模型
        restart_data = {
            "model": "./models/rwkv7-g1b-1.5b-20251202-ctx8192.pth",
            "strategy": "cuda fp16",
            "tokenizer": "",
            "customCuda": False
        }
        
        # 发送重启请求
        response = requests.post(
            f"{BACKEND_URL}/switch-model",
            headers={"Content-Type": "application/json"},
            json=restart_data
        )
        
        if response.status_code == 200:
            # 应用状态调优
            state_tuning_path = "./models/rwkv-state-tuning-NekoQA-10K-1.5B.pth"
            update_config_data = {
                "state": state_tuning_path
            }
            
            requests.post(
                f"{BACKEND_URL}/update-config",
                headers={"Content-Type": "application/json"},
                json=update_config_data
            )
            
            return jsonify({"status": "success", "message": "Model restarted successfully"})
        else:
            return jsonify({"status": "error", "message": f"Failed to restart model: {response.text}"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)