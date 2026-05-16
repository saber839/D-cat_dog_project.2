import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


class ModelHandler:
    """模型加载、推理、Grad-CAM一体化处理"""

    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 导入CNN模型（需要把cnn_model.py放到backend目录）
        from cnn_model import CNN5Layer
        self.model = CNN5Layer(num_classes=2).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.class_names = ['猫', '狗']
        self.class_names_en = ['cat', 'dog']

        # Grad-CAM相关
        self._setup_gradcam()

    def _setup_gradcam(self):
        """初始化Grad-CAM"""
        self.target_layer = self.model.conv5.conv
        self.gradients = None
        self.activations = None

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_backward_hook(save_gradient)

    def predict(self, image):
        """预测图片类别"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)

        probs = probs.squeeze().cpu().numpy()
        pred_class = int(output.argmax(dim=1).item())

        return {
            'prediction': self.class_names[pred_class],
            'prediction_en': self.class_names_en[pred_class],
            'confidence': float(probs[pred_class] * 100),
            'probabilities': {
                self.class_names[i]: float(probs[i] * 100)
                for i in range(len(self.class_names))
            }
        }

    def generate_gradcam(self, image):
        """生成Grad-CAM热力图"""
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # 前向传播
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][pred_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().detach().numpy()

        # 上采样到原图尺寸
        original_np = np.array(image.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))

        # 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加
        superimposed = np.uint8(0.5 * original_np + 0.5 * heatmap)

        return superimposed