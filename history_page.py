#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QDateEdit, QComboBox, QFileDialog, QMessageBox, QDialog,
    QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor

class DetailDialog(QDialog):
    """检测详情对话框，显示检测记录的详细信息"""
    
    def __init__(self, parent, record_data):
        super().__init__(parent)
        self.record_data = record_data
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 设置窗口属性
        self.setWindowTitle("检测详情")
        self.setMinimumSize(800, 600)
        
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 图像显示区域
        image_panel = QGroupBox("结果图像")
        image_layout = QVBoxLayout(image_panel)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        image_layout.addWidget(self.image_label)
        
        # 设置原始图像（如果有）
        if os.path.exists(self.record_data['image_path']):
            img = cv2.imread(self.record_data['image_path'])
            if img is not None:
                # 转换为Pixmap并显示
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # 缩放图像以适应标签
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("无法加载图像")
        else:
            self.image_label.setText("图像文件不存在")
        
        # 添加到分割器
        splitter.addWidget(image_panel)
        
        # 详细信息区域
        details_panel = QGroupBox("检测详情")
        details_layout = QVBoxLayout(details_panel)
        
        # 基本信息
        basic_info_group = QGroupBox("基本信息")
        basic_info_layout = QGridLayout(basic_info_group)
        
        # 添加基本信息
        basic_info_layout.addWidget(QLabel("检测时间:"), 0, 0)
        basic_info_layout.addWidget(QLabel(self.record_data['timestamp']), 0, 1)
        
        basic_info_layout.addWidget(QLabel("检测结果:"), 1, 0)
        result_label = QLabel(self.record_data['result'])
        result_color = "green" if self.record_data['result'] == "正常" else "red"
        result_label.setStyleSheet(f"color: {result_color};")
        basic_info_layout.addWidget(result_label, 1, 1)
        
        basic_info_layout.addWidget(QLabel("异常分数:"), 2, 0)
        basic_info_layout.addWidget(QLabel(f"{float(self.record_data['anomaly_score']):.4f}"), 2, 1)
        
        basic_info_layout.addWidget(QLabel("缺陷数量:"), 3, 0)
        basic_info_layout.addWidget(QLabel(str(self.record_data['defect_count'])), 3, 1)
        
        basic_info_layout.addWidget(QLabel("平均缺陷面积:"), 4, 0)
        basic_info_layout.addWidget(QLabel(f"{float(self.record_data['avg_defect_area']):.2f} 像素"), 4, 1)
        
        basic_info_layout.addWidget(QLabel("使用的阈值:"), 5, 0)
        basic_info_layout.addWidget(QLabel(f"{float(self.record_data['threshold_used']):.4f}"), 5, 1)
        
        details_layout.addWidget(basic_info_group)
        
        # 缺陷详情（如果有）
        if 'defects' in self.record_data and self.record_data['defects']:
            defect_group = QGroupBox("缺陷详情")
            defect_layout = QVBoxLayout(defect_group)
            
            # 创建表格
            defect_table = QTableWidget()
            defect_table.setColumnCount(5)
            defect_table.setHorizontalHeaderLabels(["缺陷ID", "类型", "位置", "尺寸", "面积"])
            defect_table.setRowCount(len(self.record_data['defects']))
            
            # 填充表格
            for row, defect in enumerate(self.record_data['defects']):
                defect_table.setItem(row, 0, QTableWidgetItem(str(defect['id'])))
                defect_table.setItem(row, 1, QTableWidgetItem(defect['defect_type']))
                defect_table.setItem(row, 2, QTableWidgetItem(f"({defect['x']}, {defect['y']})"))
                defect_table.setItem(row, 3, QTableWidgetItem(f"{defect['width']}x{defect['height']}"))
                defect_table.setItem(row, 4, QTableWidgetItem(f"{float(defect['area']):.2f}"))
            
            defect_layout.addWidget(defect_table)
            details_layout.addWidget(defect_group)
        
        # 添加到分割器
        splitter.addWidget(details_panel)
        
        # 设置初始分割大小
        splitter.setSizes([400, 400])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button)

class HistoryPage(QWidget):
    """历史记录页面，显示和管理检测历史记录"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent
        self.db_manager = parent.db_manager
        self.setup_ui()
        self.load_records()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建筛选面板
        filter_panel = self.create_filter_panel()
        main_layout.addWidget(filter_panel)
        
        # 创建记录表格
        self.records_table = QTableWidget()
        self.records_table.setColumnCount(7)
        self.records_table.setHorizontalHeaderLabels([
            "ID", "时间", "检测结果", "异常分数", 
            "缺陷数量", "平均缺陷面积", "操作"
        ])
        
        # 设置表格属性
        self.records_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.records_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.records_table.setAlternatingRowColors(True)
        self.records_table.setColumnWidth(0, 50)   # ID列宽
        self.records_table.setColumnWidth(1, 180)  # 时间列宽
        self.records_table.setColumnWidth(2, 80)   # 结果列宽
        self.records_table.setColumnWidth(3, 100)  # 异常分数列宽
        self.records_table.setColumnWidth(4, 80)   # 缺陷数量列宽
        self.records_table.setColumnWidth(5, 120)  # 平均缺陷面积列宽
        self.records_table.setColumnWidth(6, 150)  # 操作列宽
        
        # 添加到主布局
        main_layout.addWidget(self.records_table)
        
        # 创建状态面板
        status_layout = QHBoxLayout()
        self.status_label = QLabel("共加载 0 条记录")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        
        # 导出按钮
        self.export_btn = QPushButton("导出记录")
        self.export_btn.clicked.connect(self.export_records)
        status_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(status_layout)
    
    def create_filter_panel(self):
        """创建筛选面板"""
        panel = QGroupBox("筛选条件")
        layout = QHBoxLayout(panel)
        
        # 开始日期
        layout.addWidget(QLabel("开始日期:"))
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.setCalendarPopup(True)
        layout.addWidget(self.start_date)
        
        # 结束日期
        layout.addWidget(QLabel("结束日期:"))
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        layout.addWidget(self.end_date)
        
        # 检测结果
        layout.addWidget(QLabel("检测结果:"))
        self.result_combo = QComboBox()
        self.result_combo.addItems(["全部", "正常", "异常"])
        layout.addWidget(self.result_combo)
        
        # 筛选按钮
        self.filter_btn = QPushButton("筛选")
        self.filter_btn.clicked.connect(self.load_records)
        layout.addWidget(self.filter_btn)
        
        # 重置按钮
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_filters)
        layout.addWidget(self.reset_btn)
        
        return panel
    
    def load_records(self):
        """加载检测记录"""
        # 获取筛选条件
        start_date = datetime.combine(self.start_date.date().toPyDate(), datetime.min.time())
        end_date = datetime.combine(self.end_date.date().toPyDate(), datetime.max.time())
        result_filter = self.result_combo.currentText()
        if result_filter == "全部":
            result_filter = None
        
        # 从数据库获取记录
        records = self.db_manager.get_detection_records(
            start_date=start_date,
            end_date=end_date,
            result_filter=result_filter
        )
        
        # 清空表格
        self.records_table.setRowCount(0)
        
        # 填充表格
        for row, record in enumerate(records):
            self.records_table.insertRow(row)
            
            # ID
            self.records_table.setItem(row, 0, QTableWidgetItem(str(record['id'])))
            
            # 时间
            time_str = record['timestamp']
            self.records_table.setItem(row, 1, QTableWidgetItem(time_str))
            
            # 检测结果
            result_item = QTableWidgetItem(record['result'])
            result_color = QColor("green") if record['result'] == "正常" else QColor("red")
            result_item.setForeground(result_color)
            self.records_table.setItem(row, 2, result_item)
            
            # 异常分数
            anomaly_score = float(record['anomaly_score'])
            self.records_table.setItem(row, 3, QTableWidgetItem(f"{anomaly_score:.4f}"))
            
            # 缺陷数量
            self.records_table.setItem(row, 4, QTableWidgetItem(str(record['defect_count'])))
            
            # 平均缺陷面积
            avg_area = float(record['avg_defect_area']) if record['avg_defect_area'] is not None else 0.0
            self.records_table.setItem(row, 5, QTableWidgetItem(f"{avg_area:.2f}"))
            
            # 操作按钮
            view_btn = QPushButton("查看详情")
            view_btn.setProperty("record_id", record['id'])
            view_btn.clicked.connect(self.view_record_details)
            self.records_table.setCellWidget(row, 6, view_btn)
        
        # 更新状态标签
        self.status_label.setText(f"共加载 {len(records)} 条记录")
    
    def reset_filters(self):
        """重置筛选条件"""
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.end_date.setDate(QDate.currentDate())
        self.result_combo.setCurrentIndex(0)  # "全部"
        self.load_records()
    
    def view_record_details(self):
        """查看记录详情"""
        # 获取发送信号的按钮
        sender = self.sender()
        if not sender:
            return
        
        # 获取记录ID
        record_id = sender.property("record_id")
        if not record_id:
            return
        
        # 从数据库获取详细信息
        record_data = self.db_manager.get_record_details(record_id)
        if not record_data:
            QMessageBox.warning(self, "警告", f"无法找到ID为 {record_id} 的记录")
            return
        
        # 显示详情对话框
        dialog = DetailDialog(self, record_data)
        dialog.exec_()
    
    def export_records(self):
        """导出记录到CSV文件"""
        # 获取筛选条件
        start_date = datetime.combine(self.start_date.date().toPyDate(), datetime.min.time())
        end_date = datetime.combine(self.end_date.date().toPyDate(), datetime.max.time())
        result_filter = self.result_combo.currentText()
        if result_filter == "全部":
            result_filter = None
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出记录",
            os.path.expanduser("~/检测记录.csv"),
            "CSV文件 (*.csv)"
        )
        
        if not file_path:
            return
        
        # 导出记录
        if self.db_manager.export_records_csv(
            file_path,
            start_date=start_date,
            end_date=end_date,
            result_filter=result_filter
        ):
            QMessageBox.information(self, "成功", f"记录已导出到: {file_path}")
        else:
            QMessageBox.warning(self, "警告", "导出记录失败")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        event.accept() 