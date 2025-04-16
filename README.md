# 瓶子印刷质量检测系统

基于深度学习的瓶子印刷质量异常检测系统，专为检测各种印刷质量缺陷而设计，包括字体缺失、模糊、油污、字体变形、重影、对比度问题等。

## 项目特点

- **无监督学习**：只需训练正常样本，无需标注缺陷
- **重点检测文字区域**：对瓶子上的文字印刷区域进行重点检测
- **高精度定位**：直接用红色矩形框标注异常区域
- **批量检测能力**：支持对大量图像进行批量处理
- **可视化报告**：生成直观的异常检测报告
- **易于部署**：提供命令行和图形界面两种使用方式

## 安装说明

### 依赖环境

- Python 3.7-3.9
- PyTorch 1.7+
- CUDA (可选，推荐用于更快的训练与推理)

### 安装步骤

1. 克隆本仓库：
```bash
git clone https://github.com/yourusername/bottle_defect_detection.git
cd bottle_defect_detection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. (可选) 安装Tesseract OCR引擎（提升文字区域检测能力）：
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

## 快速入门

### 数据准备

1. 将正常瓶子图像放入`raw_images`目录
2. 运行数据预处理脚本：
```bash
python data_preparation.py
```

### 模型训练

```bash
python train_optimized.py --batch_size 16 --flow_steps 16 --region_aware --mixed_precision
```

### 检测异常

单张图像检测：
```bash
python inference.py --image_path path/to/image.jpg
```

批量图像检测：
```bash
python batch_inference.py --image_dir path/to/folder --save_images
```

### 图形界面启动

```bash
python main.py
```

## 项目结构

- `dataset/` - 训练和测试数据目录
- `raw_images/` - 原始图像存放目录
- `results/` - 训练结果和预测输出目录
- `train_optimized.py` - 优化版训练脚本
- `inference.py` - 单图检测脚本
- `batch_inference.py` - 批量检测脚本
- `data_preparation.py` - 数据预处理脚本
- `main.py` - 图形界面入口

## 版本历史

### 版本 1.0.0
- 初始版本发布
- 基本的印刷缺陷检测功能

### 版本 2.0.0
- 增加文字区域重点检测
- 添加动态阈值机制
- 改进可视化显示（红色矩形框标注）
- 增强批量处理能力
- 新增图形界面

## 常见问题

### 安装相关问题

1. **Q: 安装anomalib失败怎么办？**
   A: 尝试先安装其依赖，如`pip install pytorch-lightning==1.5.0`，然后再安装anomalib。

2. **Q: 找不到CUDA怎么办？**
   A: 系统会自动使用CPU，但训练速度会较慢。确认是否正确安装了CUDA和兼容的PyTorch版本。

### 使用相关问题

1. **Q: 什么样的图像适合训练？**
   A: 清晰、光照均匀、角度一致的正常瓶子图像，印刷质量良好。

2. **Q: 检测不准确怎么办？**
   A: 参考使用指南中的"模型调优建议"部分，调整参数或增加训练数据。

## Git使用建议

为保证顺利提交到GitHub，建议遵循以下步骤：

1. **初始提交前**：确保已经创建`.gitignore`文件，避免提交大型数据集和模型文件

2. **创建新分支进行开发**：
```bash
git checkout -b feature/new-feature
```

3. **提交代码**：
```bash
git add .
git commit -m "feat: 添加新功能描述"
```

4. **创建Release**：在GitHub界面上，点击"Releases" -> "Draft a new release"
   - 填写版本号(如v1.0.0)
   - 添加更新说明
   - 发布release

## 许可证

MIT

## 联系方式

如有问题，请通过Issues或者Pull Requests联系我们。 