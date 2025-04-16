#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境依赖检查脚本
用于验证瓶子印刷质量检测系统所需的关键依赖是否正确安装
"""

import sys
import os

def print_separator():
    print("="*50)

def check_dependency(name, import_name=None):
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        else:
            version = "未知版本"
        print(f"✓ {name}: {version}")
        return module
    except ImportError:
        print(f"✗ {name}: 未安装")
        return None

def main():
    print_separator()
    print("瓶子印刷质量检测系统 - 环境依赖检查")
    print_separator()
    
    print(f"Python: {sys.version}")
    print(f"OS: {os.name}")
    print(f"Platform: {sys.platform}")
    print_separator()
    
    print("核心依赖:")
    numpy = check_dependency("NumPy", "numpy")
    torch = check_dependency("PyTorch", "torch")
    torchvision = check_dependency("TorchVision", "torchvision")
    yaml = check_dependency("PyYAML", "yaml")
    
    print_separator()
    print("图像处理:")
    cv2 = check_dependency("OpenCV", "cv2")
    pil = check_dependency("Pillow", "PIL")
    skimage = check_dependency("scikit-image", "skimage")
    
    print_separator()
    print("机器学习:")
    sklearn = check_dependency("scikit-learn", "sklearn")
    
    print_separator()
    print("数据可视化:")
    matplotlib = check_dependency("Matplotlib", "matplotlib")
    
    print_separator()
    print("特定依赖:")
    tqdm = check_dependency("tqdm")
    pytesseract = check_dependency("pytesseract")
    
    print_separator()
    print("深度学习框架:")
    anomalib = check_dependency("anomalib")
    pytorch_lightning = check_dependency("PyTorch Lightning", "pytorch_lightning")
    
    print_separator()
    print("GUI界面:")
    try:
        import PyQt5
        from PyQt5 import QtCore
        print(f"✓ PyQt5: {QtCore.QT_VERSION_STR}")
    except ImportError:
        print("✗ PyQt5: 未安装")
    
    print_separator()
    print("CUDA可用性:")
    if torch:
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print_separator()
    
    # 检查Tesseract OCR
    print("Tesseract OCR检查:")
    if pytesseract:
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract OCR: {tesseract_version}")
        except Exception as e:
            print(f"✗ Tesseract OCR: 安装错误 - {str(e)}")
    print_separator()
    
    # 总结
    print("环境检查总结:")
    if all([numpy, torch, torchvision, yaml, cv2, pil, skimage, sklearn, 
           matplotlib, tqdm, pytesseract, anomalib, pytorch_lightning]):
        print("所有关键依赖已正确安装！")
    else:
        print("警告: 一些依赖未正确安装，请查看上面的详细信息。")
    print_separator()

if __name__ == "__main__":
    main() 