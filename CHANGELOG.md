# 更新日志

所有对项目的重要更改都将记录在此文件中。

格式基于[Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循[语义化版本 2.0.0](https://semver.org/lang/zh-CN/)规范。

## [2.0.0] - 2023-04-16

### 新增
- 增加文字区域重点检测功能，提高印刷问题检测精度
- 添加动态阈值机制，自动调整检测灵敏度
- 新增图形用户界面，支持更直观的操作
- 批量处理增强，支持批量生成详细分析报告
- 缺陷数量和面积统计功能
- 检测结果的JSON格式导出

### 修改
- 改进可视化显示方式，使用红色矩形框替代热图
- 优化训练流程，支持更多数据增强选项
- 提升模型训练速度，支持混合精度训练
- 更新依赖库版本，提高兼容性

### 修复
- 修复边缘检测不准确的问题
- 解决高分辨率图像处理缓慢的问题
- 修复某些特殊光照条件下的误检问题

## [1.0.0] - 2023-01-20

### 新增
- 首次发布
- 基于FastFlow模型的印刷缺陷检测
- 支持训练自定义模型
- 支持单图像和批量检测
- 基础数据预处理功能
- 简单的结果可视化 