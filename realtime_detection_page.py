#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import time
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QMessageBox, QFileDialog, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor

from camera_manager import CameraManager

class RealtimeDetectionPage(QWidget):
    """实时检测页面，处理摄像头实时检测瓶子印刷质量"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        
        # 获取数据库管理器和检测器
        self.db_manager = parent.db_manager
        self.detector = parent.detector
        
        # 初始化摄像头管理器
        self.camera_manager = CameraManager()
        self.camera_manager.frame_ready.connect(self.update_camera_view)
        self.camera_manager.error_occurred.connect(self.show_error)
        
        # 连接检测器信号
        self.detector.detection_completed.connect(self.handle_detection_result)
        self.detector.error_occurred.connect(self.show_error)
        
        # 设置界面
        self.setup_ui()
        
        # 初始化状态变量
        self.is_detecting = False
        self.current_result = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.perform_detection)
        
        # 加载设置
        self.load_settings()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建顶部控制区域
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 创建分割器（摄像头视图和结果显示）
        splitter = QSplitter(Qt.Horizontal)
        
        # 添加摄像头视图面板
        camera_panel = self.create_camera_panel()
        splitter.addWidget(camera_panel)
        
        # 添加结果显示面板
        results_panel = self.create_results_panel()
        splitter.addWidget(results_panel)
        
        # 设置分割器初始大小
        splitter.setSizes([600, 400])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter, 1)  # 分配伸展因子
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("控制面板")
        layout = QHBoxLayout(panel)
        
        # 摄像头控制
        camera_control_group = QGroupBox("摄像头控制")
        camera_control_layout = QGridLayout(camera_control_group)
        
        # 摄像头ID选择
        camera_control_layout.addWidget(QLabel("摄像头ID:"), 0, 0)
        self.camera_id_spinbox = QSpinBox()
        self.camera_id_spinbox.setRange(0, 10)
        self.camera_id_spinbox.setValue(0)
        camera_control_layout.addWidget(self.camera_id_spinbox, 0, 1)
        
        # 摄像头开关按钮
        self.camera_toggle_btn = QPushButton("启动摄像头")
        self.camera_toggle_btn.clicked.connect(self.toggle_camera)
        camera_control_layout.addWidget(self.camera_toggle_btn, 0, 2)
        
        layout.addWidget(camera_control_group)
        
        # 检测控制
        detection_control_group = QGroupBox("检测控制")
        detection_control_layout = QGridLayout(detection_control_group)
        
        # 检测间隔
        detection_control_layout.addWidget(QLabel("检测间隔(秒):"), 0, 0)
        self.detection_interval_spinbox = QDoubleSpinBox()
        self.detection_interval_spinbox.setRange(0.1, 10.0)
        self.detection_interval_spinbox.setValue(1.0)
        self.detection_interval_spinbox.setSingleStep(0.1)
        detection_control_layout.addWidget(self.detection_interval_spinbox, 0, 1)
        
        # 检测开关按钮
        self.detection_toggle_btn = QPushButton("开始检测")
        self.detection_toggle_btn.clicked.connect(self.toggle_detection)
        self.detection_toggle_btn.setEnabled(False)  # 初始禁用，等摄像头启动
        detection_control_layout.addWidget(self.detection_toggle_btn, 0, 2)
        
        layout.addWidget(detection_control_group)
        
        # 保存控制
        save_control_group = QGroupBox("保存控制")
        save_control_layout = QGridLayout(save_control_group)
        
        # 自动保存复选框
        self.auto_save_checkbox = QCheckBox("自动保存异常结果")
        self.auto_save_checkbox.setChecked(True)
        save_control_layout.addWidget(self.auto_save_checkbox, 0, 0, 1, 2)
        
        # 保存路径按钮
        self.save_path_btn = QPushButton("选择保存路径")
        self.save_path_btn.clicked.connect(self.select_save_path)
        save_control_layout.addWidget(self.save_path_btn, 0, 2)
        
        layout.addWidget(save_control_group)
        
        return panel
    
    def create_camera_panel(self):
        """创建摄像头显示面板"""
        panel = QGroupBox("实时摄像头")
        layout = QVBoxLayout(panel)
        
        # 摄像头视图
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setStyleSheet("background-color: #000;")
        self.camera_view.setText("等待摄像头启动...")
        layout.addWidget(self.camera_view, 1)  # 分配伸展因子
        
        # 状态显示
        self.camera_status_label = QLabel("摄像头状态: 未启动")
        layout.addWidget(self.camera_status_label)
        
        return panel
    
    def create_results_panel(self):
        """创建结果显示面板"""
        panel = QGroupBox("检测结果")
        layout = QVBoxLayout(panel)
        
        # 结果标题
        self.result_title_label = QLabel("等待检测...")
        self.result_title_label.setAlignment(Qt.AlignCenter)
        self.result_title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(self.result_title_label)
        
        # 结果详情
        result_details_group = QGroupBox("详细信息")
        result_details_layout = QGridLayout(result_details_group)
        
        # 异常分数
        result_details_layout.addWidget(QLabel("异常分数:"), 0, 0)
        self.anomaly_score_label = QLabel("-")
        result_details_layout.addWidget(self.anomaly_score_label, 0, 1)
        
        # 缺陷数量
        result_details_layout.addWidget(QLabel("缺陷数量:"), 1, 0)
        self.defect_count_label = QLabel("-")
        result_details_layout.addWidget(self.defect_count_label, 1, 1)
        
        # 平均缺陷面积
        result_details_layout.addWidget(QLabel("平均缺陷面积:"), 2, 0)
        self.avg_defect_area_label = QLabel("-")
        result_details_layout.addWidget(self.avg_defect_area_label, 2, 1)
        
        # 使用的阈值
        result_details_layout.addWidget(QLabel("使用的阈值:"), 3, 0)
        self.threshold_label = QLabel("-")
        result_details_layout.addWidget(self.threshold_label, 3, 1)
        
        layout.addWidget(result_details_group)
        
        # 缺陷列表
        defect_list_group = QGroupBox("缺陷列表")
        defect_list_layout = QVBoxLayout(defect_list_group)
        
        self.defect_list_label = QLabel("无缺陷")
        defect_list_layout.addWidget(self.defect_list_label)
        
        layout.addWidget(defect_list_group)
        
        # 手动保存按钮
        self.save_result_btn = QPushButton("保存当前结果")
        self.save_result_btn.clicked.connect(self.save_current_result)
        self.save_result_btn.setEnabled(False)
        layout.addWidget(self.save_result_btn)
        
        # 自动扩展
        layout.addStretch(1)
        
        return panel
    
    def toggle_camera(self):
        """切换摄像头状态"""
        if self.camera_manager.is_camera_running():
            # 停止摄像头
            self.stop_camera()
        else:
            # 启动摄像头
            self.start_camera()
    
    def start_camera(self):
        """启动摄像头"""
        # 获取摄像头ID
        camera_id = self.camera_id_spinbox.value()
        
        # 设置摄像头ID
        self.camera_manager.set_camera_id(camera_id)
        
        # 启动摄像头
        if self.camera_manager.start_camera():
            # 更新UI
            self.camera_toggle_btn.setText("停止摄像头")
            self.camera_status_label.setText(f"摄像头状态: 已启动 (ID: {camera_id})")
            self.detection_toggle_btn.setEnabled(True)
            
            # 保存设置
            self.db_manager.save_setting('camera_id', str(camera_id))
        else:
            QMessageBox.critical(self, "错误", f"无法启动摄像头 ID: {camera_id}")
    
    def stop_camera(self):
        """停止摄像头"""
        # 如果正在检测，先停止检测
        if self.is_detecting:
            self.toggle_detection()
        
        # 停止摄像头
        if self.camera_manager.stop_camera():
            # 更新UI
            self.camera_toggle_btn.setText("启动摄像头")
            self.camera_status_label.setText("摄像头状态: 已停止")
            self.detection_toggle_btn.setEnabled(False)
            
            # 清空摄像头视图
            blank_pixmap = QPixmap(640, 480)
            blank_pixmap.fill(Qt.black)
            self.camera_view.setPixmap(blank_pixmap)
            self.camera_view.setText("等待摄像头启动...")
    
    def toggle_detection(self):
        """切换检测状态"""
        if self.is_detecting:
            # 停止检测
            self.detection_timer.stop()
            self.is_detecting = False
            self.detection_toggle_btn.setText("开始检测")
        else:
            # 确保模型已加载
            if not self.detector.ensure_model_loaded():
                QMessageBox.critical(self, "错误", "无法加载检测模型，请检查模型配置")
                return
            
            # 开始检测
            interval_ms = int(self.detection_interval_spinbox.value() * 1000)
            self.detection_timer.start(interval_ms)
            self.is_detecting = True
            self.detection_toggle_btn.setText("停止检测")
    
    def perform_detection(self):
        """执行检测"""
        # 获取当前帧
        current_frame = self.camera_manager.get_latest_frame()
        
        if current_frame is None:
            return
        
        # 进行检测
        self.detector.detect_image(current_frame)
    
    def update_camera_view(self, pixmap):
        """更新摄像头视图
        
        参数:
            pixmap (QPixmap): 要显示的图像
        """
        # 调整图像大小以适应窗口
        scaled_pixmap = pixmap.scaled(
            self.camera_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 更新图像
        self.camera_view.setPixmap(scaled_pixmap)
    
    def handle_detection_result(self, result):
        """处理检测结果
        
        参数:
            result (dict): 检测结果
        """
        # 保存当前结果
        self.current_result = result
        
        # 更新结果显示
        self.update_result_display(result)
        
        # 如果需要自动保存异常结果
        if self.auto_save_checkbox.isChecked() and result['result'] == "异常":
            self.save_result_to_database(result)
    
    def update_result_display(self, result):
        """更新结果显示
        
        参数:
            result (dict): 检测结果
        """
        # 更新结果标题
        is_normal = result['result'] == "正常"
        title_text = "检测结果: " + result['result']
        title_color = "green" if is_normal else "red"
        self.result_title_label.setText(f"<span style='color:{title_color};'>{title_text}</span>")
        
        # 更新结果详情
        self.anomaly_score_label.setText(f"{result['anomaly_score']:.4f}")
        self.defect_count_label.setText(str(result['defect_count']))
        self.avg_defect_area_label.setText(f"{result['avg_defect_area']:.2f} 像素")
        self.threshold_label.setText(f"{result['threshold_used']:.4f}")
        
        # 更新缺陷列表
        if result['defect_count'] > 0:
            defect_text = "<ul>"
            for i, defect in enumerate(result['defects']):
                defect_text += f"<li>缺陷 {i+1}: 位置 ({defect['x']}, {defect['y']}), 大小: {defect['width']}x{defect['height']}, 面积: {defect['area']:.2f}</li>"
            defect_text += "</ul>"
            self.defect_list_label.setText(defect_text)
        else:
            self.defect_list_label.setText("无缺陷")
        
        # 启用保存按钮
        self.save_result_btn.setEnabled(True)
    
    def save_result_to_database(self, result):
        """保存结果到数据库
        
        参数:
            result (dict): 检测结果
        """
        try:
            # 生成保存路径
            save_dir = self.db_manager.get_setting('image_save_path', 'results/images')
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{timestamp}_{result['result']}.jpg"
            image_path = os.path.join(save_dir, image_filename)
            
            # 保存图像
            self.detector.save_result_image(result, image_path)
            
            # 创建数据库记录
            record_data = {
                'image_path': image_path,
                'result': result['result'],
                'anomaly_score': result['anomaly_score'],
                'defect_count': result['defect_count'],
                'avg_defect_area': result['avg_defect_area'],
                'threshold_used': result['threshold_used'],
                'defects': result['defects']
            }
            
            # 保存到数据库
            self.db_manager.save_detection_result(record_data)
            
        except Exception as e:
            self.show_error(f"保存结果失败: {str(e)}")
    
    def save_current_result(self):
        """保存当前结果"""
        if self.current_result is None:
            return
        
        self.save_result_to_database(self.current_result)
        QMessageBox.information(self, "成功", "检测结果已保存")
    
    def select_save_path(self):
        """选择保存路径"""
        current_path = self.db_manager.get_setting('image_save_path', 'results/images')
        
        # 打开文件对话框
        new_path = QFileDialog.getExistingDirectory(
            self,
            "选择保存路径",
            current_path
        )
        
        if new_path:
            # 保存新路径
            self.db_manager.save_setting('image_save_path', new_path)
    
    def load_settings(self):
        """加载设置"""
        # 加载摄像头ID
        camera_id = int(self.db_manager.get_setting('camera_id', '0'))
        self.camera_id_spinbox.setValue(camera_id)
    
    def show_error(self, message):
        """显示错误消息
        
        参数:
            message (str): 错误消息
        """
        QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止摄像头
        if self.camera_manager.is_camera_running():
            self.camera_manager.stop_camera() 