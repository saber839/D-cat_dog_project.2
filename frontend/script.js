// ==================== 后端API配置 ====================
const API_BASE_URL = 'http://localhost:5000/api';

// ==================== DOM元素 ====================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const fileCount = document.getElementById('fileCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const resultGrid = document.getElementById('resultGrid');
const detailModal = document.getElementById('detailModal');
const modalBody = document.getElementById('modalBody');

// Agent相关DOM元素
const agentSection = document.getElementById('agentSection');
const agentMessage = document.getElementById('agentMessage');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const resetChatBtn = document.getElementById('resetChatBtn');
const llmBadge = document.getElementById('llmBadge');

// 存储选择的文件
let selectedFiles = [];

// ==================== 文件上传逻辑 ====================
uploadArea.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// 拖拽上传
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    const newFiles = Array.from(files).filter(f => validTypes.includes(f.type));

    if (newFiles.length === 0) {
        alert('请选择 JPG、PNG 或 BMP 格式的图片');
        return;
    }

    selectedFiles = [...selectedFiles, ...newFiles];
    renderPreview();
}

function renderPreview() {
    if (selectedFiles.length === 0) {
        previewSection.style.display = 'none';
        return;
    }

    previewSection.style.display = 'block';
    fileCount.textContent = selectedFiles.length;
    previewGrid.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}" alt="预览图片 ${index + 1}">
                <button class="remove-btn" data-index="${index}">×</button>
            `;
            previewGrid.appendChild(div);

            div.querySelector('.remove-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                selectedFiles.splice(index, 1);
                renderPreview();
            });
        };
        reader.readAsDataURL(file);
    });
}

clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    agentSection.style.display = 'none';
});

// ==================== Agent分析 ====================
async function analyzeWithAgent(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch(`${API_BASE_URL}/agent/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const data = await response.json();

        // 显示Agent面板和分析报告
        agentSection.style.display = 'block';
        agentMessage.textContent = data.agent_response;

        // 同时显示预测结果卡片
        displayResults([{
            prediction: data.prediction,
            confidence: data.confidence,
            original_image: data.original_image,
            gradcam_image: data.gradcam_image,
            probabilities: data.probabilities
        }]);

        // 滚动到Agent面板
        agentSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Agent分析失败:', error);
        agentSection.style.display = 'block';
        agentMessage.textContent = '分析失败，请检查后端服务是否启动。';
    }
}

analyzeBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        alert('请先选择图片');
        return;
    }

    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';

    await analyzeWithAgent(selectedFiles[0]);

    loadingSection.style.display = 'none';
});

// ==================== 结果展示 ====================
function displayResults(results) {
    resultSection.style.display = 'block';
    resultGrid.innerHTML = '';

    results.forEach((result, index) => {
        if (result.error) {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <p style="color: var(--danger);">❌ ${result.filename || '图片'}: ${result.error}</p>
            `;
            resultGrid.appendChild(card);
            return;
        }

        const predClass = result.prediction_en ||
                         (result.prediction === '猫' ? 'cat' : 'dog');

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="card-image">
                <img src="${result.original_image}" alt="原图">
            </div>
            <div class="card-info">
                <div class="prediction-label ${predClass}">
                    ${result.prediction === '猫' ? '🐱' : '🐶'} ${result.prediction}
                </div>
                <div class="confidence">
                    置信度: ${result.confidence.toFixed(2)}%
                </div>
                <div class="confidence-bar">
                    <div class="fill ${predClass}"
                         style="width: ${result.confidence}%"></div>
                </div>
            </div>
        `;

        card.addEventListener('click', () => showDetail(result));
        resultGrid.appendChild(card);
    });
}

// ==================== Agent对话 ====================
async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    agentMessage.textContent = '思考中...';

    try {
        const response = await fetch(`${API_BASE_URL}/agent/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        agentMessage.textContent = data.response;
        chatInput.value = '';

    } catch (error) {
        console.error('对话失败:', error);
        agentMessage.textContent = '对话失败，请检查后端服务是否启动。';
    }
}

async function resetChat() {
    try {
        await fetch(`${API_BASE_URL}/agent/reset`, { method: 'POST' });
        agentMessage.textContent = '对话已重置。请上传新图片或提出新问题。';
        chatInput.value = '';
    } catch (error) {
        console.error('重置失败:', error);
    }
}

async function checkAgentStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.llm_enabled) {
            llmBadge.textContent = 'LLM增强';
            llmBadge.style.background = '#e6ffe6';
            llmBadge.style.color = '#28a745';
        }
    } catch (error) {
        console.warn('Agent状态检查失败');
    }
}

sendBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});
resetChatBtn.addEventListener('click', resetChat);

// ==================== 详情弹窗 ====================
function showDetail(result) {
    const predClass = result.prediction_en ||
                     (result.prediction === '猫' ? 'cat' : 'dog');

    modalBody.innerHTML = `
        <h3 style="color: ${predClass === 'cat' ? '#e67e22' : '#2980b9'}; text-align: center;">
            ${result.prediction === '猫' ? '🐱' : '🐶'} ${result.prediction}
        </h3>
        <p style="text-align: center; font-size: 1.2rem; margin-bottom: 20px;">
            置信度: <strong>${result.confidence.toFixed(2)}%</strong>
        </p>

        <div class="modal-images">
            <div>
                <h4 style="text-align: center; margin-bottom: 10px;">📷 原始图片</h4>
                <img src="${result.original_image}" alt="原图">
            </div>
            <div>
                <h4 style="text-align: center; margin-bottom: 10px;">🔥 Grad-CAM 热力图</h4>
                <img src="${result.gradcam_image}" alt="Grad-CAM">
            </div>
        </div>

        <div style="margin-top: 20px;">
            <h4 style="margin-bottom: 10px;">各类别概率:</h4>
            ${Object.entries(result.probabilities)
                .map(([cls, prob]) => `
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="width: 50px; font-weight: 500;">${cls}</span>
                        <div style="flex: 1; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 0 10px;">
                            <div style="height: 100%; width: ${prob}%;
                                 background: ${cls === '猫' ? 'linear-gradient(90deg, #f39c12, #e67e22)' : 'linear-gradient(90deg, #3498db, #2980b9)'};
                                 border-radius: 10px; transition: width 0.5s ease;"></div>
                        </div>
                        <span style="width: 60px; text-align: right;">${prob.toFixed(2)}%</span>
                    </div>
                `).join('')}
        </div>

        <p style="margin-top: 20px; padding: 15px; background: #f0f5ff; border-radius: 8px; font-size: 0.9rem; color: #666;">
            💡 <strong>Grad-CAM 说明：</strong>红色区域表示模型做出分类决策时最关注的图像区域。
        </p>
    `;

    detailModal.classList.add('active');
}

document.querySelector('.modal-close').addEventListener('click', () => {
    detailModal.classList.remove('active');
});

detailModal.addEventListener('click', (e) => {
    if (e.target === detailModal) {
        detailModal.classList.remove('active');
    }
});

// ==================== 健康检查 ====================
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('后端服务状态:', data.message);
    } catch {
        console.warn('后端服务未启动，请运行 backend/app.py');
    }
}

// ==================== 页面初始化 ====================
checkHealth();
checkAgentStatus();