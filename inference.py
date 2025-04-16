import os
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytesseract
from pathlib import Path
from anomalib.models.fastflow import FastflowModel

def load_model(config_path):
    """
    加载已训练的模型
    
    参数:
        config_path: 模型配置文件路径
    
    返回:
        加载的模型
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 加载模型检查点
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型检查点: {checkpoint_path}")
    
    model = FastflowModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device, config["input_size"]

def preprocess_image(image_path, input_size=448):
    """
    预处理测试图像
    
    参数:
        image_path: 图像路径
        input_size: 模型输入尺寸
    
    返回:
        处理后的图像张量和原始图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 转换为RGB并调整大小
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (input_size, input_size))
    
    # 转换为PyTorch张量并归一化
    image_tensor = torch.from_numpy(resized_image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor, original_image

def detect_text_regions(image, min_text_area=100):
    """
    使用OCR检测图像中的文字区域
    
    参数:
        image: 输入RGB图像
        min_text_area: 最小文字区域面积
    
    返回:
        文字区域掩码
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 自适应阈值二值化，突出文字区域
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作：闭运算，连接相邻文字
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建文字区域掩码
    h, w = image.shape[:2]
    text_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 筛选可能的文字区域轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_text_area:
            # 绘制轮廓到掩码上
            cv2.drawContours(text_mask, [contour], 0, 255, -1)
    
    # 如果没有检测到文字区域，返回全图掩码
    if np.sum(text_mask) == 0:
        text_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    return text_mask

def detect_anomaly(model, image_tensor, device, threshold=0.5):
    """
    使用模型检测图像中的异常
    
    参数:
        model: 加载的模型
        image_tensor: 预处理后的图像张量
        device: 运行设备
        threshold: 异常分数阈值
    
    返回:
        anomaly_map: 异常热力图
        anomaly_score: 异常分数
    """
    # 将张量移动到相应设备
    image_tensor = image_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 获取异常分数和异常图
    anomaly_map = outputs["anomaly_map"]
    anomaly_score = outputs["pred_score"].cpu().item()
    
    # 处理异常图
    anomaly_map = anomaly_map.cpu().numpy()
    
    return anomaly_map, anomaly_score

def dynamic_threshold(anomaly_score):
    """
    根据异常分数动态调整阈值
    
    参数:
        anomaly_score: 异常分数
        
    返回:
        threshold_ratio: 阈值比例
    """
    # 根据异常分数调整阈值
    if anomaly_score > 0.8:  # 严重异常
        return 0.5  # 较低阈值，检测更多可能的异常区域
    elif anomaly_score > 0.6:  # 中等异常
        return 0.6
    elif anomaly_score > 0.5:  # 轻微异常
        return 0.7
    else:  # 可能正常
        return 0.8  # 较高阈值，减少误报

def visualize_results(original_image, anomaly_map, anomaly_score, output_path=None, threshold_ratio=None):
    """
    可视化异常检测结果，仅用矩形框标注缺陷区域，重点关注文字区域
    
    参数:
        original_image: 原始图像
        anomaly_map: 异常热力图
        anomaly_score: 异常分数
        output_path: 结果保存路径
        threshold_ratio: 阈值比例，用于二值化异常图，如果为None则使用动态阈值
    """
    # 检测文字区域
    text_mask = detect_text_regions(original_image)
    
    # 动态确定阈值
    if threshold_ratio is None:
        threshold_ratio = dynamic_threshold(anomaly_score)
    
    # 将异常图调整为与原始图像相同尺寸
    h, w = original_image.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map[0, 0], (w, h))
    
    # 归一化异常图到0-1，然后缩放到0-255
    anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
    anomaly_map_norm_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
    
    # 计算阈值，大于阈值的认为是异常区域
    threshold_value = threshold_ratio * anomaly_map_norm.max()
    _, binary_map = cv2.threshold(anomaly_map_norm_uint8, int(threshold_value * 255), 255, cv2.THRESH_BINARY)
    
    # 将文字区域掩码应用到二值图上，重点检测文字区域的异常
    binary_map = cv2.bitwise_and(binary_map, text_mask)
    
    # 形态学操作，连接相邻异常区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    
    # 寻找轮廓并在原图上标记
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anomaly_image = original_image.copy()
    
    # 绘制轮廓，将异常区域用红色矩形框标注
    min_contour_area = 10  # 降低最小轮廓面积以检测更小的缺陷
    has_anomaly = False
    defect_count = 0
    defect_areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            has_anomaly = True
            defect_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            defect_areas.append(area)
            cv2.rectangle(anomaly_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 红色框
    
    # 创建可视化图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(original_image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    title = f"检测结果 (分数: {anomaly_score:.4f}, 阈值: {threshold_ratio:.2f})"
    if defect_count > 0:
        title += f", 检测到{defect_count}处缺陷"
    plt.title(title)
    plt.imshow(anomaly_image)
    
    # 如果有缺陷，在图上标注"异常"，否则标注"正常"
    if has_anomaly and anomaly_score > 0.5:
        status = "异常"
        color = 'red'
    else:
        status = "正常"
        color = 'green'
    
    plt.text(10, 30, status, fontsize=16, color=color, weight='bold', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"结果已保存到: {output_path}")
    
    plt.show()
    
    # 返回异常区域信息
    anomaly_info = {
        "has_anomaly": has_anomaly and anomaly_score > 0.5,
        "anomaly_score": anomaly_score,
        "defect_count": defect_count,
        "defect_areas": defect_areas,
        "threshold_used": threshold_ratio
    }
    
    return anomaly_image, anomaly_info

def process_image(image_path, model_config_path, output_dir="results/predictions", threshold_ratio=None):
    """
    处理单张图像并可视化结果
    
    参数:
        image_path: 图像路径
        model_config_path: 模型配置文件路径
        output_dir: 输出目录
        threshold_ratio: 阈值比例，如果为None则使用动态阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model, device, input_size = load_model(model_config_path)
    
    # 预处理图像
    print("预处理图像...")
    image_tensor, original_image = preprocess_image(image_path, input_size)
    
    # 检测异常
    print("检测异常...")
    anomaly_map, anomaly_score = detect_anomaly(model, image_tensor, device)
    
    # 结果保存路径
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_anomaly_result.png")
    
    # 可视化结果
    print("可视化结果...")
    _, anomaly_info = visualize_results(original_image, anomaly_map, anomaly_score, output_path, threshold_ratio)
    
    # 打印异常信息
    print(f"\n图像 '{image_path}' 检测结果:")
    print(f"异常分数: {anomaly_score:.4f}")
    print(f"检测状态: {'异常' if anomaly_info['has_anomaly'] else '正常'}")
    
    if anomaly_info['defect_count'] > 0:
        print(f"缺陷数量: {anomaly_info['defect_count']}")
        avg_defect_size = sum(anomaly_info['defect_areas']) / len(anomaly_info['defect_areas'])
        print(f"平均缺陷面积: {avg_defect_size:.1f}像素")
    
    print(f"使用的阈值比例: {anomaly_info['threshold_used']:.2f}")
    
    return anomaly_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastFlow模型推理")
    parser.add_argument("--image_path", type=str, required=True, help="测试图像路径")
    parser.add_argument("--config_path", type=str, default="results/model_config.yaml", help="模型配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results/predictions", help="结果输出目录")
    parser.add_argument("--threshold_ratio", type=float, default=None, 
                        help="异常区域阈值比例，不指定则使用动态阈值")
    
    args = parser.parse_args()
    
    process_image(
        image_path=args.image_path,
        model_config_path=args.config_path,
        output_dir=args.output_dir,
        threshold_ratio=args.threshold_ratio
    ) 