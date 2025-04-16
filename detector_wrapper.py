#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal

class DetectorWrapper(QObject):
    """检测器包装类，包装瓶子印刷质量检测模型，提供易用的接口"""
    
    # 定义信号
    detection_completed = pyqtSignal(dict)  # 检测完成信号，传递检测结果
    error_occurred = pyqtSignal(str)        # 错误信号，传递错误信息
    
    def __init__(self, model_path="results/model_config.yaml"):
        """初始化检测器
        
        参数:
            model_path (str): 模型配置文件路径
        """
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.device = None
        self.input_size = None
        self.is_loaded = False
        self.infer_functions = {}  # 存储导入的推理函数
    
    def load_model(self):
        """加载模型
        
        返回:
            bool: 加载是否成功
        """
        try:
            # 导入必要的函数
            import sys
            if not os.path.exists('inference.py'):
                self.error_occurred.emit("找不到inference.py文件，请确保当前工作目录正确")
                return False
            
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            # 从inference模块导入函数
            from inference import (
                load_model, 
                preprocess_image, 
                detect_anomaly, 
                detect_text_regions,
                dynamic_threshold,
                visualize_results
            )
            
            # 存储推理函数
            self.infer_functions = {
                'load_model': load_model,
                'preprocess_image': preprocess_image,
                'detect_anomaly': detect_anomaly,
                'detect_text_regions': detect_text_regions,
                'dynamic_threshold': dynamic_threshold,
                'visualize_results': visualize_results
            }
            
            # 加载模型
            self.model, self.device, self.input_size = load_model(self.model_path)
            
            # 标记模型已加载
            self.is_loaded = True
            return True
            
        except ImportError as e:
            self.error_occurred.emit(f"导入推理模块失败: {str(e)}")
            return False
        except Exception as e:
            self.error_occurred.emit(f"加载模型失败: {str(e)}")
            return False
    
    def ensure_model_loaded(self):
        """确保模型已加载
        
        返回:
            bool: 模型是否已加载
        """
        if not self.is_loaded:
            return self.load_model()
        return True
    
    def detect_image(self, image, save_path=None):
        """检测图像中的瓶子印刷质量
        
        参数:
            image (numpy.ndarray): 输入图像，BGR格式(OpenCV)
            save_path (str): 结果保存路径，None表示不保存
            
        返回:
            dict: 检测结果，包含以下键:
                - result_image: 可视化结果图像
                - anomaly_score: 异常分数
                - result: 检测结果 ("正常"/"异常")
                - defect_count: 缺陷数量
                - avg_defect_area: 平均缺陷面积
                - defects: 缺陷列表
                - threshold_used: 使用的阈值
        """
        # 确保模型已加载
        if not self.ensure_model_loaded():
            return None
        
        try:
            # 获取推理函数
            preprocess_image = self.infer_functions['preprocess_image']
            detect_anomaly = self.infer_functions['detect_anomaly']
            detect_text_regions = self.infer_functions['detect_text_regions']
            dynamic_threshold = self.infer_functions['dynamic_threshold']
            visualize_results = self.infer_functions['visualize_results']
            
            # 图像预处理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测文本区域
            text_regions = detect_text_regions(image_rgb)
            
            # 预处理图像
            input_tensor = preprocess_image(image_rgb, self.input_size)
            
            # 使用模型检测异常
            anomaly_map, anomaly_score = detect_anomaly(self.model, input_tensor, self.device)
            
            # 动态阈值
            threshold_ratio = dynamic_threshold(anomaly_score)
            
            # 可视化结果
            result_image, anomaly_info = visualize_results(
                image_rgb, 
                anomaly_map, 
                anomaly_score,
                text_regions=text_regions,
                output_path=save_path,
                threshold_ratio=threshold_ratio
            )
            
            # 获取缺陷信息
            defect_count = anomaly_info.get('defect_count', 0)
            avg_defect_area = anomaly_info.get('avg_defect_area', 0.0)
            defects = anomaly_info.get('defects', [])
            
            # 判断结果
            result = "正常" if defect_count == 0 else "异常"
            
            # 构建结果字典
            result_dict = {
                'result_image': cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR),  # 转回BGR格式
                'anomaly_score': float(anomaly_score),
                'result': result,
                'defect_count': defect_count,
                'avg_defect_area': avg_defect_area,
                'defects': defects,
                'threshold_used': threshold_ratio
            }
            
            # 发出检测完成信号
            self.detection_completed.emit(result_dict)
            
            return result_dict
            
        except Exception as e:
            self.error_occurred.emit(f"检测过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_camera_frame(self, frame):
        """处理摄像头帧
        
        参数:
            frame (numpy.ndarray): 摄像头帧，BGR格式
            
        返回:
            numpy.ndarray: 处理后的帧，带有检测结果
        """
        # 进行检测
        result = self.detect_image(frame)
        
        if result is None:
            # 检测失败，返回原始帧
            return frame
        
        # 返回带有检测结果的帧
        return result['result_image']
    
    def save_result_image(self, result_dict, image_path):
        """保存结果图像
        
        参数:
            result_dict (dict): 检测结果字典
            image_path (str): 图像保存路径
            
        返回:
            bool: 保存是否成功
        """
        try:
            if 'result_image' in result_dict:
                # 创建目录（如果不存在）
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                # 保存图像
                cv2.imwrite(image_path, result_dict['result_image'])
                return True
            return False
        except Exception as e:
            self.error_occurred.emit(f"保存结果图像失败: {str(e)}")
            return False 