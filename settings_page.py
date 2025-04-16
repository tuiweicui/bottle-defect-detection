#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QMessageBox, QFileDialog, QTabWidget, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class SettingsPage(QWidget):
    """系统设置页面，管理系统各项设置"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.db_manager = parent.db_manager
        self.detector = parent.detector
        
        # 原始设置值，用于检测是否修改
        self.original_settings = {}
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建标签页控件
        tabs = QTabWidget()
        
        # 添加常规设置标签页
        general_tab = self.create_general_settings()
        tabs.addTab(general_tab, "常规设置")
        
        # 添加检测设置标签页
        detection_tab = self.create_detection_settings()
        tabs.addTab(detection_tab, "检测设置")
        
        # 添加摄像头设置标签页
        camera_tab = self.create_camera_settings()
        tabs.addTab(camera_tab, "摄像头设置")
        
        # 添加标签页控件到主布局
        main_layout.addWidget(tabs)
        
        # 添加操作按钮
        buttons_layout = QHBoxLayout()
        
        # 保存按钮
        self.save_btn = QPushButton("保存设置")
        self.save_btn.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("重置设置")
        self.reset_btn.clicked.connect(self.load_settings)
        buttons_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(buttons_layout)
    
    def create_general_settings(self):
        """创建常规设置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 应用设置组
        app_group = QGroupBox("应用设置")
        app_layout = QGridLayout(app_group)
        
        # 保存路径
        app_layout.addWidget(QLabel("检测结果保存路径:"), 0, 0)
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setReadOnly(True)
        app_layout.addWidget(self.save_path_edit, 0, 1)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_save_path)
        app_layout.addWidget(self.browse_btn, 0, 2)
        
        # 自动保存检测结果
        self.auto_save_checkbox = QCheckBox("自动保存异常检测结果")
        app_layout.addWidget(self.auto_save_checkbox, 1, 0, 1, 3)
        
        # 应用启动设置
        self.auto_load_model_checkbox = QCheckBox("应用启动时自动加载模型")
        app_layout.addWidget(self.auto_load_model_checkbox, 2, 0, 1, 3)
        
        layout.addWidget(app_group)
        
        # 数据库设置组
        db_group = QGroupBox("数据库设置")
        db_layout = QGridLayout(db_group)
        
        # 数据库路径
        db_layout.addWidget(QLabel("数据库文件路径:"), 0, 0)
        self.db_path_edit = QLineEdit()
        self.db_path_edit.setReadOnly(True)
        db_layout.addWidget(self.db_path_edit, 0, 1)
        
        # 清理按钮
        self.clean_db_btn = QPushButton("清理数据库")
        self.clean_db_btn.clicked.connect(self.confirm_clean_database)
        db_layout.addWidget(self.clean_db_btn, 1, 0, 1, 2)
        
        layout.addWidget(db_group)
        
        # 添加伸展因子
        layout.addStretch(1)
        
        return panel
    
    def create_detection_settings(self):
        """创建检测设置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QGridLayout(model_group)
        
        # 模型路径
        model_layout.addWidget(QLabel("模型配置文件路径:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        
        self.model_browse_btn = QPushButton("浏览...")
        self.model_browse_btn.clicked.connect(self.browse_model_path)
        model_layout.addWidget(self.model_browse_btn, 0, 2)
        
        # 重新加载模型按钮
        self.reload_model_btn = QPushButton("重新加载模型")
        self.reload_model_btn.clicked.connect(self.reload_model)
        model_layout.addWidget(self.reload_model_btn, 1, 0, 1, 3)
        
        layout.addWidget(model_group)
        
        # 检测阈值设置组
        threshold_group = QGroupBox("检测阈值设置")
        threshold_layout = QGridLayout(threshold_group)
        
        # 基础阈值
        threshold_layout.addWidget(QLabel("基础阈值系数:"), 0, 0)
        self.base_threshold_spinbox = QDoubleSpinBox()
        self.base_threshold_spinbox.setRange(0.01, 1.0)
        self.base_threshold_spinbox.setSingleStep(0.01)
        self.base_threshold_spinbox.setDecimals(2)
        threshold_layout.addWidget(self.base_threshold_spinbox, 0, 1)
        
        # 最小缺陷面积
        threshold_layout.addWidget(QLabel("最小缺陷面积(像素):"), 1, 0)
        self.min_defect_area_spinbox = QSpinBox()
        self.min_defect_area_spinbox.setRange(1, 10000)
        threshold_layout.addWidget(self.min_defect_area_spinbox, 1, 1)
        
        # 使用动态阈值
        self.dynamic_threshold_checkbox = QCheckBox("使用动态阈值")
        threshold_layout.addWidget(self.dynamic_threshold_checkbox, 2, 0, 1, 2)
        
        # 检测文本区域
        self.detect_text_regions_checkbox = QCheckBox("检测文本区域")
        threshold_layout.addWidget(self.detect_text_regions_checkbox, 3, 0, 1, 2)
        
        layout.addWidget(threshold_group)
        
        # 添加伸展因子
        layout.addStretch(1)
        
        return panel
    
    def create_camera_settings(self):
        """创建摄像头设置面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 摄像头设置组
        camera_group = QGroupBox("摄像头设置")
        camera_layout = QGridLayout(camera_group)
        
        # 摄像头ID
        camera_layout.addWidget(QLabel("默认摄像头ID:"), 0, 0)
        self.camera_id_spinbox = QSpinBox()
        self.camera_id_spinbox.setRange(0, 10)
        camera_layout.addWidget(self.camera_id_spinbox, 0, 1)
        
        # 分辨率设置
        camera_layout.addWidget(QLabel("摄像头分辨率:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480", "800x600", "1024x768", "1280x720", "1920x1080"
        ])
        camera_layout.addWidget(self.resolution_combo, 1, 1)
        
        # 帧率设置
        camera_layout.addWidget(QLabel("目标帧率:"), 2, 0)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        camera_layout.addWidget(self.fps_spinbox, 2, 1)
        
        # 自动检测间隔
        camera_layout.addWidget(QLabel("自动检测间隔(秒):"), 3, 0)
        self.detection_interval_spinbox = QDoubleSpinBox()
        self.detection_interval_spinbox.setRange(0.1, 10.0)
        self.detection_interval_spinbox.setSingleStep(0.1)
        self.detection_interval_spinbox.setDecimals(1)
        camera_layout.addWidget(self.detection_interval_spinbox, 3, 1)
        
        layout.addWidget(camera_group)
        
        # 添加伸展因子
        layout.addStretch(1)
        
        return panel
    
    def load_settings(self):
        """从数据库加载设置"""
        # 常规设置
        self.save_path_edit.setText(self.db_manager.get_setting('image_save_path', 'results/images'))
        self.auto_save_checkbox.setChecked(self.db_manager.get_setting('save_images', 'true') == 'true')
        self.auto_load_model_checkbox.setChecked(self.db_manager.get_setting('auto_load_model', 'true') == 'true')
        self.db_path_edit.setText(self.db_manager.db_path)
        
        # 检测设置
        self.model_path_edit.setText(self.db_manager.get_setting('model_path', 'results/model_config.yaml'))
        self.base_threshold_spinbox.setValue(float(self.db_manager.get_setting('detection_threshold', '0.5')))
        self.min_defect_area_spinbox.setValue(int(self.db_manager.get_setting('min_defect_area', '100')))
        self.dynamic_threshold_checkbox.setChecked(self.db_manager.get_setting('dynamic_threshold', 'true') == 'true')
        self.detect_text_regions_checkbox.setChecked(self.db_manager.get_setting('detect_text_regions', 'true') == 'true')
        
        # 摄像头设置
        self.camera_id_spinbox.setValue(int(self.db_manager.get_setting('camera_id', '0')))
        
        # 分辨率
        resolution = self.db_manager.get_setting('camera_resolution', '640x480')
        index = self.resolution_combo.findText(resolution)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)
        
        # 帧率
        self.fps_spinbox.setValue(int(self.db_manager.get_setting('camera_fps', '30')))
        
        # 检测间隔
        self.detection_interval_spinbox.setValue(float(self.db_manager.get_setting('detection_interval', '1.0')))
        
        # 保存原始设置，用于检测修改
        self.original_settings = {
            'image_save_path': self.save_path_edit.text(),
            'save_images': self.auto_save_checkbox.isChecked(),
            'auto_load_model': self.auto_load_model_checkbox.isChecked(),
            'model_path': self.model_path_edit.text(),
            'detection_threshold': self.base_threshold_spinbox.value(),
            'min_defect_area': self.min_defect_area_spinbox.value(),
            'dynamic_threshold': self.dynamic_threshold_checkbox.isChecked(),
            'detect_text_regions': self.detect_text_regions_checkbox.isChecked(),
            'camera_id': self.camera_id_spinbox.value(),
            'camera_resolution': self.resolution_combo.currentText(),
            'camera_fps': self.fps_spinbox.value(),
            'detection_interval': self.detection_interval_spinbox.value()
        }
    
    def save_settings(self):
        """保存设置到数据库"""
        try:
            # 常规设置
            self.db_manager.save_setting('image_save_path', self.save_path_edit.text())
            self.db_manager.save_setting('save_images', 'true' if self.auto_save_checkbox.isChecked() else 'false')
            self.db_manager.save_setting('auto_load_model', 'true' if self.auto_load_model_checkbox.isChecked() else 'false')
            
            # 检测设置
            self.db_manager.save_setting('model_path', self.model_path_edit.text())
            self.db_manager.save_setting('detection_threshold', str(self.base_threshold_spinbox.value()))
            self.db_manager.save_setting('min_defect_area', str(self.min_defect_area_spinbox.value()))
            self.db_manager.save_setting('dynamic_threshold', 'true' if self.dynamic_threshold_checkbox.isChecked() else 'false')
            self.db_manager.save_setting('detect_text_regions', 'true' if self.detect_text_regions_checkbox.isChecked() else 'false')
            
            # 摄像头设置
            self.db_manager.save_setting('camera_id', str(self.camera_id_spinbox.value()))
            self.db_manager.save_setting('camera_resolution', self.resolution_combo.currentText())
            self.db_manager.save_setting('camera_fps', str(self.fps_spinbox.value()))
            self.db_manager.save_setting('detection_interval', str(self.detection_interval_spinbox.value()))
            
            # 更新原始设置
            self.load_settings()
            
            QMessageBox.information(self, "成功", "设置已保存")
            
            # 检查是否需要重新加载模型
            if (self.original_settings['model_path'] != self.model_path_edit.text() and 
                self.detector.is_loaded):
                reply = QMessageBox.question(
                    self, 
                    "重新加载模型", 
                    "模型路径已更改，是否立即重新加载模型？",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.reload_model()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
    
    def browse_save_path(self):
        """浏览并选择保存路径"""
        current_path = self.save_path_edit.text()
        
        # 如果目录不存在，使用当前目录
        if not os.path.exists(current_path):
            current_path = os.getcwd()
        
        # 打开文件对话框
        new_path = QFileDialog.getExistingDirectory(
            self,
            "选择保存路径",
            current_path
        )
        
        if new_path:
            self.save_path_edit.setText(new_path)
    
    def browse_model_path(self):
        """浏览并选择模型路径"""
        current_path = self.model_path_edit.text()
        
        # 获取当前目录
        current_dir = os.path.dirname(current_path) if os.path.exists(current_path) else os.getcwd()
        
        # 打开文件对话框
        new_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型配置文件",
            current_dir,
            "YAML文件 (*.yaml);;所有文件 (*.*)"
        )
        
        if new_path:
            self.model_path_edit.setText(new_path)
    
    def reload_model(self):
        """重新加载模型"""
        try:
            # 更新模型路径
            new_model_path = self.model_path_edit.text()
            self.detector.model_path = new_model_path
            
            # 重置加载状态
            self.detector.is_loaded = False
            
            # 尝试加载模型
            if self.detector.load_model():
                QMessageBox.information(self, "成功", "模型已重新加载")
            else:
                QMessageBox.critical(self, "错误", "模型加载失败，请检查模型配置文件")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新加载模型失败: {str(e)}")
    
    def confirm_clean_database(self):
        """确认清理数据库"""
        reply = QMessageBox.question(
            self, 
            "清理数据库", 
            "确定要清理数据库吗？这将删除所有历史记录！",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.clean_database()
    
    def clean_database(self):
        """清理数据库"""
        try:
            # 假设数据库管理器有一个clean_records方法
            if hasattr(self.db_manager, 'clean_records') and callable(self.db_manager.clean_records):
                self.db_manager.clean_records()
                QMessageBox.information(self, "成功", "数据库已清理")
            else:
                QMessageBox.warning(self, "警告", "清理功能未实现")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"清理数据库失败: {str(e)}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 检查是否有未保存的修改
        if self.has_unsaved_changes():
            reply = QMessageBox.question(
                self, 
                "未保存的修改", 
                "有未保存的设置更改，是否保存？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, 
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.save_settings()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        event.accept()
    
    def has_unsaved_changes(self):
        """检查是否有未保存的更改"""
        # 检查各设置项是否有变化
        if (
            self.original_settings['image_save_path'] != self.save_path_edit.text() or
            self.original_settings['save_images'] != self.auto_save_checkbox.isChecked() or
            self.original_settings['auto_load_model'] != self.auto_load_model_checkbox.isChecked() or
            self.original_settings['model_path'] != self.model_path_edit.text() or
            self.original_settings['detection_threshold'] != self.base_threshold_spinbox.value() or
            self.original_settings['min_defect_area'] != self.min_defect_area_spinbox.value() or
            self.original_settings['dynamic_threshold'] != self.dynamic_threshold_checkbox.isChecked() or
            self.original_settings['detect_text_regions'] != self.detect_text_regions_checkbox.isChecked() or
            self.original_settings['camera_id'] != self.camera_id_spinbox.value() or
            self.original_settings['camera_resolution'] != self.resolution_combo.currentText() or
            self.original_settings['camera_fps'] != self.fps_spinbox.value() or
            self.original_settings['detection_interval'] != self.detection_interval_spinbox.value()
        ):
            return True
        
        return False 