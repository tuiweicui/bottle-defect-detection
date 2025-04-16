#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                            QListWidget, QStackedWidget, QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from realtime_detection_page import RealtimeDetectionPage
from history_page import HistoryPage
from settings_page import SettingsPage
from database_manager import DatabaseManager
from detector_wrapper import DetectorWrapper

class MainWindow(QMainWindow):
    """瓶子印刷质量检测系统主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化窗口属性
        self.setWindowTitle("瓶子印刷质量检测系统")
        self.setGeometry(100, 100, 1280, 720)
        
        # 初始化数据库连接
        self.db_manager = DatabaseManager()
        
        # 初始化检测器
        self.detector = DetectorWrapper()
        
        # 设置UI
        self.setup_ui()
        
        # 显示状态栏
        self.statusBar().showMessage("系统就绪")
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧导航栏
        self.create_navigation_panel(main_layout)
        
        # 创建右侧堆叠窗口
        self.create_stacked_pages(main_layout)
    
    def create_navigation_panel(self, parent_layout):
        """创建左侧导航面板"""
        # 创建导航列表
        self.nav_list = QListWidget()
        self.nav_list.addItem("实时检测")
        self.nav_list.addItem("历史记录")
        self.nav_list.addItem("系统设置")
        
        # 设置导航列表属性
        self.nav_list.setFixedWidth(150)
        self.nav_list.setIconSize(Qt.QSize(24, 24))
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: #2c3e50;
                border: none;
                outline: none;
                color: #ecf0f1;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #34495e;
            }
            QListWidget::item:selected {
                background-color: #3498db;
            }
        """)
        
        # 连接信号到槽
        self.nav_list.currentRowChanged.connect(self.change_page)
        
        # 添加到父布局
        parent_layout.addWidget(self.nav_list)
    
    def create_stacked_pages(self, parent_layout):
        """创建右侧堆叠页面"""
        # 创建堆叠窗口部件
        self.stack_widget = QStackedWidget()
        
        # 创建各个页面
        self.realtime_page = RealtimeDetectionPage(self)
        self.history_page = HistoryPage(self)
        self.settings_page = SettingsPage(self)
        
        # 添加页面到堆叠窗口
        self.stack_widget.addWidget(self.realtime_page)
        self.stack_widget.addWidget(self.history_page)
        self.stack_widget.addWidget(self.settings_page)
        
        # 添加到父布局
        parent_layout.addWidget(self.stack_widget)
    
    def change_page(self, index):
        """切换页面"""
        self.stack_widget.setCurrentIndex(index)
        
        # 更新状态栏信息
        page_names = ["实时检测", "历史记录", "系统设置"]
        self.statusBar().showMessage(f"当前页面: {page_names[index]}")
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 提示用户确认关闭
        reply = QMessageBox.question(
            self, 
            '确认退出', 
            '确定要退出瓶子印刷质量检测系统吗?',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 关闭摄像头等资源
            if hasattr(self.realtime_page, 'camera_manager') and self.realtime_page.camera_manager:
                self.realtime_page.camera_manager.stop_camera()
            
            # 关闭数据库连接
            if self.db_manager:
                self.db_manager.close_connection()
                
            event.accept()
        else:
            event.ignore() 