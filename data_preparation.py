import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_directory_structure():
    """创建MVTec格式的目录结构"""
    os.makedirs("dataset/bottle/train/good", exist_ok=True)
    os.makedirs("dataset/bottle/test/good", exist_ok=True)
    os.makedirs("dataset/bottle/val/good", exist_ok=True)  # 添加验证集目录

def simulate_text_quality_issues(image):
    """模拟文本打印质量问题的数据增强"""
    h, w = image.shape[:2]
    result = image.copy()
    
    # 随机选择一种文本质量问题进行模拟
    issue_type = random.choice(['blur', 'missing', 'stain', 'double', 'deform', 'low_contrast'])
    
    if issue_type == 'blur':
        # 模拟字体模糊：局部高斯模糊或运动模糊
        if random.random() < 0.5:
            # 整体轻微模糊
            blur_level = random.randint(1, 5) * 2 + 1
            result = cv2.GaussianBlur(result, (blur_level, blur_level), 0)
        else:
            # 局部区域模糊
            x1, y1 = random.randint(0, w//3), random.randint(0, h//3)
            x2, y2 = x1 + random.randint(w//3, w//2), y1 + random.randint(h//3, h//2)
            roi = result[y1:y2, x1:x2]
            blur_level = random.randint(3, 9) * 2 + 1
            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_level, blur_level), 0)
            
    elif issue_type == 'missing':
        # 模拟字体缺失：局部区域用背景色填充或者随机遮挡
        for _ in range(random.randint(1, 3)):
            # 创建小块随机遮挡
            mask_w, mask_h = random.randint(5, w//10), random.randint(5, h//10)
            mask_x, mask_y = random.randint(0, w-mask_w), random.randint(0, h-mask_h)
            
            # 用随机颜色或背景色填充
            if random.random() < 0.5:
                # 背景色填充（假设为白色或浅色）
                result[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = [255, 255, 255]
            else:
                # 随机遮挡
                mask_color = np.random.randint(180, 256, 3).tolist()
                result[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = mask_color
            
    elif issue_type == 'stain':
        # 模拟油污、污渍
        for _ in range(random.randint(2, 5)):
            # 创建随机形状的污渍
            center_x, center_y = random.randint(0, w), random.randint(0, h)
            axes_length = (random.randint(5, 20), random.randint(5, 20))
            angle = random.randint(0, 180)
            stain_color = np.random.randint(0, 100, 3).tolist()  # 深色污渍
            
            # 创建污渍蒙版
            stain_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(stain_mask, (center_x, center_y), axes_length, angle, 0, 360, 255, -1)
            
            # 应用半透明污渍
            alpha = random.uniform(0.3, 0.7)  # 透明度
            for c in range(3):
                result[:, :, c] = np.where(
                    stain_mask == 255, 
                    result[:, :, c] * (1 - alpha) + stain_color[c] * alpha, 
                    result[:, :, c]
                )
                
    elif issue_type == 'double':
        # 模拟重影、重叠打印
        shift_x, shift_y = random.randint(1, 5), random.randint(1, 5)
        alpha = random.uniform(0.2, 0.4)  # 重影透明度
        
        # 创建偏移副本
        shifted = np.zeros_like(result)
        shifted[shift_y:, shift_x:] = result[:-shift_y, :-shift_x] if shift_y > 0 and shift_x > 0 else result
        
        # 叠加原图和偏移图
        result = cv2.addWeighted(result, 1.0, shifted, alpha, 0)
        
    elif issue_type == 'deform':
        # 模拟字体变形
        # 使用随机变形网格
        grid_size = 10
        mesh_scale = random.uniform(5, 15)
        
        mapx = np.zeros((h, w), dtype=np.float32)
        mapy = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                mapx[i:i+grid_size, j:j+grid_size] = j + random.uniform(-mesh_scale, mesh_scale)
                mapy[i:i+grid_size, j:j+grid_size] = i + random.uniform(-mesh_scale, mesh_scale)
        
        for i in range(h):
            for j in range(w):
                if mapx[i, j] < 0:
                    mapx[i, j] = 0
                if mapx[i, j] >= w:
                    mapx[i, j] = w - 1
                if mapy[i, j] < 0:
                    mapy[i, j] = 0
                if mapy[i, j] >= h:
                    mapy[i, j] = h - 1
        
        result = cv2.remap(result, mapx, mapy, cv2.INTER_LINEAR)
        
    elif issue_type == 'low_contrast':
        # 模拟字体不清晰、对比度低
        alpha = random.uniform(0.6, 0.9)  # 降低对比度
        beta = random.randint(10, 30)  # 增加亮度
        
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        # 可以添加轻微噪点
        noise = np.zeros(result.shape, np.uint8)
        cv2.randn(noise, 0, random.randint(5, 15))
        result = cv2.add(result, noise)
    
    return result

def apply_augmentation(image):
    """应用数据增强方法生成更多样本"""
    augmented_images = []
    
    # 原始图像
    augmented_images.append(image)
    
    # 基础数据增强
    # 旋转
    for angle in [90, 180, 270]:
        rotated = cv2.rotate(image, 
                             {90: cv2.ROTATE_90_CLOCKWISE,
                              180: cv2.ROTATE_180,
                              270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        augmented_images.append(rotated)
    
    # 水平和垂直翻转
    augmented_images.append(cv2.flip(image, 1))  # 水平翻转
    augmented_images.append(cv2.flip(image, 0))  # 垂直翻转
    
    # 高级数据增强
    # 亮度和对比度调整（更大范围）
    for alpha in [0.7, 0.8, 1.2, 1.3]:  # 对比度
        for beta in [-20, -10, 10, 20]:  # 亮度
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented_images.append(adjusted)
    
    # 随机裁剪再缩放回原尺寸
    h, w = image.shape[:2]
    for _ in range(5):  # 增加裁剪次数
        crop_ratio = random.uniform(0.7, 0.9)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # 随机裁剪位置
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        cropped = image[y:y+crop_h, x:x+crop_w]
        resized = cv2.resize(cropped, (w, h))
        augmented_images.append(resized)
    
    # 高斯噪声（多种程度）
    for sigma in [5, 10, 15]:
        noise = np.zeros(image.shape, np.uint8)
        cv2.randn(noise, 0, sigma)
        noisy = cv2.add(image, noise)
        augmented_images.append(noisy)
    
    # 模拟印刷质量问题
    for _ in range(8):  # 生成多个印刷质量问题样本
        text_issue = simulate_text_quality_issues(image)
        augmented_images.append(text_issue)
    
    # 组合增强：先裁剪再添加质量问题
    for _ in range(4):
        # 随机裁剪
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        
        cropped = image[y:y+crop_h, x:x+crop_w]
        resized = cv2.resize(cropped, (w, h))
        
        # 添加质量问题
        text_issue = simulate_text_quality_issues(resized)
        augmented_images.append(text_issue)
    
    return augmented_images

def preprocess_images(source_folder, target_size=(448, 448)):
    """
    预处理图像并进行数据增强
    
    参数:
    - source_folder: 包含原始图像的文件夹
    - target_size: 目标图像尺寸，默认为(448, 448)，提高分辨率以保留更多细节
    """
    # 创建目录结构
    create_directory_structure()
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(Path(source_folder).glob(ext)))
    
    if not image_files:
        print("未找到图像文件，请检查源文件夹路径!")
        return
    
    print(f"找到 {len(image_files)} 张原始图像")
    
    # 将数据分为训练集、验证集和测试集
    train_files, test_val_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)
    
    print(f"数据集划分: 训练集 {len(train_files)}张, 验证集 {len(val_files)}张, 测试集 {len(test_files)}张")
    
    # 处理训练集图像
    train_augmented_count = 0
    for idx, img_path in enumerate(train_files):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 调整图像大小
            image = cv2.resize(image, target_size)
            
            # 应用数据增强
            augmented_images = apply_augmentation(image)
            
            # 保存增强后的图像
            for aug_idx, aug_img in enumerate(augmented_images):
                output_path = f"dataset/bottle/train/good/train_{idx}_{aug_idx}.png"
                cv2.imwrite(output_path, aug_img)
                train_augmented_count += 1
                
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {str(e)}")
    
    # 处理验证集图像
    val_count = 0
    for idx, img_path in enumerate(val_files):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 调整图像大小
            image = cv2.resize(image, target_size)
            
            # 保存验证集图像
            output_path = f"dataset/bottle/val/good/val_{idx}.png"
            cv2.imwrite(output_path, image)
            val_count += 1
                
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {str(e)}")
    
    # 处理测试集图像
    test_count = 0
    for idx, img_path in enumerate(test_files):
        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 调整图像大小
            image = cv2.resize(image, target_size)
            
            # 保存原始测试图像
            output_path = f"dataset/bottle/test/good/test_{idx}.png"
            cv2.imwrite(output_path, image)
            test_count += 1
                
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {str(e)}")
    
    print(f"已处理并增强训练图像: {train_augmented_count} 张")
    print(f"已处理验证集图像: {val_count} 张")
    print(f"已处理测试集图像: {test_count} 张")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="瓶子印刷图像数据预处理与增强")
    parser.add_argument("--source_folder", type=str, default="raw_images", 
                        help="包含原始图像的文件夹路径")
    parser.add_argument("--target_size", type=int, nargs=2, default=[448, 448], 
                        help="调整后的图像尺寸，格式为'宽 高'，提高为448x448以保留更多细节")
    
    args = parser.parse_args()
    
    # 创建原始图像文件夹（如果不存在）
    os.makedirs(args.source_folder, exist_ok=True)
    
    print(f"请将您的原始正常瓶子图像放入 '{args.source_folder}' 文件夹")
    print(f"然后运行此脚本进行数据预处理和增强")
    
    # 如果文件夹中已有图像，则进行处理
    image_count = len(list(Path(args.source_folder).glob("*.jpg"))) + \
                  len(list(Path(args.source_folder).glob("*.jpeg"))) + \
                  len(list(Path(args.source_folder).glob("*.png"))) + \
                  len(list(Path(args.source_folder).glob("*.bmp")))
    
    if image_count > 0:
        print(f"在 '{args.source_folder}' 中发现 {image_count} 张图像，开始处理...")
        preprocess_images(args.source_folder, tuple(args.target_size))
    else:
        print(f"警告：'{args.source_folder}' 中没有发现图像文件") 