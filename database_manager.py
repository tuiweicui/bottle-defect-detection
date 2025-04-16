#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sqlite3
import datetime
from PyQt5.QtCore import QObject

class DatabaseManager(QObject):
    """数据库管理类，处理瓶子检测系统的数据存储和检索"""
    
    def __init__(self, db_path="bottle_defect.db"):
        super().__init__()
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """初始化数据库连接并创建必要的表结构"""
        create_new = not os.path.exists(self.db_path)
        
        # 创建数据库连接
        self.conn = sqlite3.connect(self.db_path)
        
        # 启用外键约束
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # 若是新数据库，创建表结构
        if create_new:
            self.create_tables()
        
    def create_tables(self):
        """创建数据库表结构"""
        cursor = self.conn.cursor()
        
        # 创建检测记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            result TEXT,
            anomaly_score REAL,
            defect_count INTEGER,
            avg_defect_area REAL,
            threshold_used REAL
        )
        ''')
        
        # 创建缺陷详情表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS defect_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            defect_type TEXT,
            x INTEGER,
            y INTEGER,
            width INTEGER,
            height INTEGER,
            area REAL,
            FOREIGN KEY (record_id) REFERENCES detection_records (id) ON DELETE CASCADE
        )
        ''')
        
        # 创建设置表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE,
            setting_value TEXT
        )
        ''')
        
        # 添加一些默认设置
        default_settings = [
            ('camera_id', '0'),
            ('detection_threshold', '0.5'),
            ('min_defect_area', '100'),
            ('save_images', 'true'),
            ('image_save_path', 'results/images')
        ]
        
        cursor.executemany(
            'INSERT OR IGNORE INTO settings (setting_name, setting_value) VALUES (?, ?)',
            default_settings
        )
        
        self.conn.commit()
    
    def save_detection_result(self, result_data):
        """保存检测结果到数据库
        
        参数:
            result_data (dict): 包含检测结果的字典，需包含以下键:
                - image_path: 图像路径
                - result: 检测结果 ("正常"/"异常")
                - anomaly_score: 异常分数
                - defect_count: 缺陷数量
                - avg_defect_area: 平均缺陷面积
                - threshold_used: 使用的阈值
                - defects: 缺陷详情列表，每个缺陷是一个字典
        
        返回:
            int: 新创建的记录ID
        """
        cursor = self.conn.cursor()
        
        # 保存检测记录
        cursor.execute('''
        INSERT INTO detection_records 
        (image_path, result, anomaly_score, defect_count, avg_defect_area, threshold_used)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result_data.get('image_path', ''),
            result_data.get('result', '未知'),
            result_data.get('anomaly_score', 0.0),
            result_data.get('defect_count', 0),
            result_data.get('avg_defect_area', 0.0),
            result_data.get('threshold_used', 0.0)
        ))
        
        record_id = cursor.lastrowid
        
        # 保存缺陷详情
        for defect in result_data.get('defects', []):
            cursor.execute('''
            INSERT INTO defect_details
            (record_id, defect_type, x, y, width, height, area)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id,
                defect.get('type', '未知'),
                defect.get('x', 0),
                defect.get('y', 0),
                defect.get('width', 0),
                defect.get('height', 0),
                defect.get('area', 0.0)
            ))
        
        self.conn.commit()
        return record_id
    
    def get_detection_records(self, start_date=None, end_date=None, result_filter=None, limit=100):
        """获取检测记录列表
        
        参数:
            start_date (datetime): 开始日期
            end_date (datetime): 结束日期
            result_filter (str): 结果筛选 ("正常"/"异常"/None)
            limit (int): 限制返回的记录数量
            
        返回:
            list: 检测记录列表
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM detection_records WHERE 1=1"
        params = []
        
        # 添加日期筛选
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d 00:00:00"))
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.strftime("%Y-%m-%d 23:59:59"))
        
        # 添加结果筛选
        if result_filter and result_filter != "全部":
            query += " AND result = ?"
            params.append(result_filter)
        
        # 添加排序和限制
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        # 转换为字典列表
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in records]
    
    def get_record_details(self, record_id):
        """获取指定记录的详细信息
        
        参数:
            record_id (int): 记录ID
            
        返回:
            dict: 包含记录详情和缺陷列表的字典
        """
        cursor = self.conn.cursor()
        
        # 获取记录信息
        cursor.execute("SELECT * FROM detection_records WHERE id = ?", (record_id,))
        record = cursor.fetchone()
        
        if not record:
            return None
        
        # 转换为字典
        columns = [column[0] for column in cursor.description]
        record_dict = dict(zip(columns, record))
        
        # 获取缺陷详情
        cursor.execute("SELECT * FROM defect_details WHERE record_id = ?", (record_id,))
        defects = cursor.fetchall()
        
        # 转换缺陷详情为字典列表
        defect_columns = [column[0] for column in cursor.description]
        defect_list = [dict(zip(defect_columns, defect)) for defect in defects]
        
        # 添加缺陷列表到记录字典
        record_dict['defects'] = defect_list
        
        return record_dict
    
    def get_setting(self, setting_name, default_value=None):
        """获取设置值
        
        参数:
            setting_name (str): 设置名称
            default_value: 默认值，如果设置不存在
            
        返回:
            str: 设置值
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT setting_value FROM settings WHERE setting_name = ?", (setting_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        return default_value
    
    def save_setting(self, setting_name, setting_value):
        """保存设置值
        
        参数:
            setting_name (str): 设置名称
            setting_value: 设置值
            
        返回:
            bool: 操作是否成功
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO settings (setting_name, setting_value) VALUES (?, ?)",
                (setting_name, str(setting_value))
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"保存设置失败: {str(e)}")
            return False
    
    def export_records_csv(self, file_path, start_date=None, end_date=None, result_filter=None):
        """导出检测记录到CSV文件
        
        参数:
            file_path (str): CSV文件路径
            start_date (datetime): 开始日期
            end_date (datetime): 结束日期
            result_filter (str): 结果筛选 ("正常"/"异常"/None)
            
        返回:
            bool: 操作是否成功
        """
        try:
            import csv
            
            # 获取记录
            records = self.get_detection_records(
                start_date=start_date,
                end_date=end_date,
                result_filter=result_filter,
                limit=10000  # 导出时放宽限制
            )
            
            if not records:
                return False
                
            # 写入CSV
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                # 获取字段名
                fieldnames = records[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                writer.writerows(records)
                
            return True
        except Exception as e:
            print(f"导出CSV失败: {str(e)}")
            return False
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None 