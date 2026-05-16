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

// 存储选择的文件
let selectedFiles = [];

// ==================== 文件上传逻辑 ====================
// 点击上传区域
uploadArea.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

// 文件选择
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

// 处理文件
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

// 渲染预览
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

            // 删除按钮事件
            div.querySelector('.remove-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                selectedFiles.splice(index, 1);
                renderPreview();
            });
        };
        reader.readAsDataURL(file);
    });
}

// 清空选择
clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
});

// ==================== 发起预测请求 ====================
analyzeBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        alert('请先选择图片');
        return;
    }

    // 显示加载状态
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });

    try {
        const response = await fetch(`${API_BASE_URL}/predict_batch`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`服务器错误: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data.results);
    } catch (error) {
        alert(`识别失败: ${error.message}`);
        console.error('Error:', error);
    } finally {
        loadingSection.style.display = 'none';
    }
});

// 单张预测（用于示例图片）
async function predictSingle(file) {
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        displayResults([data]);
    } catch (error) {
        alert(`识别失败: ${error.message}`);
    } finally {
        loadingSection.style.display = 'none';
    }
}

// ==================== 结果展示 ====================
function displayResults(results) {
    resultSection.style.display = 'block';
    resultGrid.innerHTML = '';

    results.forEach((result, index) => {
        if (result.error) {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <p style="color: var(--danger);">❌ ${result.filename}: ${result.error}</p>
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

        // 点击查看详情
        card.addEventListener('click', () => showDetail(result));

        resultGrid.appendChild(card);
    });
}

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
            如果红色集中在猫/狗的耳朵、面部等关键部位，说明模型学习到了正确的特征。
        </p>
    `;

    detailModal.classList.add('active');
}

// 关闭弹窗
document.querySelector('.modal-close').addEventListener('click', () => {
    detailModal.classList.remove('active');
});

detailModal.addEventListener('click', (e) => {
    if (e.target === detailModal) {
        detailModal.classList.remove('active');
    }
});

// ==================== 示例图片 ====================
document.querySelectorAll('.demo-card').forEach(card => {
    card.addEventListener('click', async () => {
        const src = card.dataset.src;
        try {
            const response = await fetch(src);
            const blob = await response.blob();
            const file = new File([blob], src, { type: 'image/jpeg' });
            predictSingle(file);
        } catch {
            alert('示例图片加载失败，请上传你自己的图片试试吧');
        }
    });
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

// 页面加载时检查
checkHealth();