#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap

class CameraManager(QObject):
    """摄像头管理类，处理摄像头的启动、停止和帧处理"""
    
    # 定义信号
    frame_ready = pyqtSignal(QPixmap)  # 帧准备好的信号，传递QPixmap
    error_occurred = pyqtSignal(str)   # 错误信号，传递错误信息
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.camera_id = 0
        self.is_running = False
        self.camera_thread = None
        self.mutex = QMutex()  # 互斥锁，保护多线程访问
        self.frame_processor = None    # 帧处理函数
        self.frame_rate = 30           # 目标帧率
        self.frame_width = 640         # 帧宽度
        self.frame_height = 480        # 帧高度
        self.latest_frame = None       # 最新处理的帧
    
    def set_camera_id(self, camera_id):
        """设置摄像头ID
        
        参数:
            camera_id (int): 摄像头ID
        """
        self.camera_id = camera_id
    
    def set_frame_processor(self, processor_func):
        """设置帧处理函数
        
        参数:
            processor_func (callable): 帧处理函数，接收numpy数组格式的帧，返回处理后的帧
        """
        with QMutexLocker(self.mutex):
            self.frame_processor = processor_func
    
    def start_camera(self):
        """启动摄像头
        
        返回:
            bool: 操作是否成功
        """
        # 检查是否已经在运行
        if self.is_running:
            return True
        
        # 尝试打开摄像头
        with QMutexLocker(self.mutex):
            self.camera = cv2.VideoCapture(self.camera_id)
            
            # 检查是否成功打开
            if not self.camera.isOpened():
                self.error_occurred.emit(f"无法打开摄像头 ID: {self.camera_id}")
                self.camera = None
                return False
            
            # 设置摄像头属性
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # 标记为运行中
            self.is_running = True
        
        # 创建并启动摄像头线程
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        return True
    
    def stop_camera(self):
        """停止摄像头
        
        返回:
            bool: 操作是否成功
        """
        # 检查是否在运行
        if not self.is_running:
            return True
        
        # 标记为停止
        self.is_running = False
        
        # 等待线程结束
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
        
        # 释放摄像头资源
        with QMutexLocker(self.mutex):
            if self.camera:
                self.camera.release()
                self.camera = None
        
        return True
    
    def is_camera_running(self):
        """检查摄像头是否在运行
        
        返回:
            bool: 摄像头是否在运行
        """
        return self.is_running
    
    def get_latest_frame(self):
        """获取最新的帧
        
        返回:
            numpy.ndarray: 最新的帧，如果没有则返回None
        """
        with QMutexLocker(self.mutex):
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def _camera_loop(self):
        """摄像头循环，在单独的线程中运行"""
        frame_time = 1.0 / self.frame_rate
        
        while self.is_running:
            start_time = time.time()
            
            # 读取摄像头帧
            with QMutexLocker(self.mutex):
                if not self.camera or not self.camera.isOpened():
                    # 摄像头已关闭或出错
                    self.error_occurred.emit("摄像头连接已断开")
                    self.is_running = False
                    break
                
                ret, frame = self.camera.read()
            
            if not ret:
                # 读取帧失败
                continue
            
            # 处理帧
            processed_frame = frame
            if self.frame_processor:
                try:
                    processed_frame = self.frame_processor(frame)
                except Exception as e:
                    self.error_occurred.emit(f"帧处理错误: {str(e)}")
                    processed_frame = frame
            
            # 保存最新的帧
            with QMutexLocker(self.mutex):
                self.latest_frame = processed_frame.copy()
            
            # 将OpenCV BGR图像转换为QPixmap
            pixmap = self._convert_cv_to_pixmap(processed_frame)
            
            # 发出帧准备好的信号
            self.frame_ready.emit(pixmap)
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _convert_cv_to_pixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap
        
        参数:
            cv_img (numpy.ndarray): OpenCV格式的图像
            
        返回:
            QPixmap: 转换后的QPixmap
        """
        # 转换颜色空间从BGR到RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 获取图像尺寸和格式
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # 创建QImage
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 转换为QPixmap并返回
        return QPixmap.fromImage(q_img)
    
    def set_camera_parameters(self, width=None, height=None, fps=None):
        """设置摄像头参数
        
        参数:
            width (int): 帧宽度
            height (int): 帧高度
            fps (int): 帧率
            
        返回:
            bool: 操作是否成功
        """
        # 更新参数
        if width:
            self.frame_width = width
        if height:
            self.frame_height = height
        if fps:
            self.frame_rate = fps
        
        # 如果摄像头已运行，应用新参数
        with QMutexLocker(self.mutex):
            if self.camera and self.camera.isOpened():
                if width:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if fps:
                    self.camera.set(cv2.CAP_PROP_FPS, fps)
                
                return True
        
        return False 