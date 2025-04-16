#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

def main():
    """程序入口点"""
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle('Fusion')
    # 设置中文字体
    from PyQt5.QtGui import QFont
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序事件循环
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 