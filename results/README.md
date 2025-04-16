# 结果目录

此目录用于存放瓶子印刷质量检测系统的训练结果、模型文件和预测输出。

## 目录结构

训练和预测过程中，系统会在此目录下创建以下结构：

```
results/
├── model_config.yaml         # 模型配置信息
├── best_model.pth            # 训练得到的最佳模型权重
├── last_checkpoint.pth       # 最后一次训练检查点
├── training_log.csv          # 训练过程记录
├── predictions/              # 单图预测结果目录
│   ├── image_001_result.png  # 单图预测可视化结果
│   ├── image_002_result.png
│   └── ...
└── batch_predictions/        # 批量预测结果目录
    ├── normal/               # 正常图像结果
    ├── anomaly/              # 异常图像结果
    ├── anomaly_detection_results.csv   # 详细结果表格
    ├── anomaly_detection_report.png    # 图形化报告
    └── anomaly_detection_report.txt    # 文本报告
```

## 模型训练结果

训练过程会保存以下文件：

1. **best_model.pth**：验证损失最低的模型权重文件
2. **last_checkpoint.pth**：最后一个epoch的检查点，可用于恢复训练
3. **model_config.yaml**：包含模型架构、超参数等配置信息
4. **training_log.csv**：记录每个epoch的训练损失、验证损失等指标

## 预测结果

1. **单图预测**（`predictions/`目录）：
   - 图像文件名为`原文件名_result.png`
   - 包含原图、异常热图和缺陷标注等可视化内容

2. **批量预测**（`batch_predictions/`目录）：
   - 正常/异常图像分别保存在不同文件夹
   - CSV表格包含详细的异常分数和统计信息
   - 图形化报告提供整体分析和统计图表
   - 文本报告提供详细的检测结果描述 