import os
import argparse
import yaml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from anomalib.data.mvtec import MVTecDataModule
from anomalib.models.fastflow import FastflowModel
from anomalib.data.utils import ValDataLoader

class LitFastflowModel(FastflowModel):
    """
    增强版FastFlow模型，加入学习率调度功能
    """
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

def train_model(
    data_path, 
    category="bottle",
    backbone="wide_resnet50_2",  # 使用更强大的骨干网络
    batch_size=32,
    num_workers=4,
    max_epochs=200,  # 增加训练轮数
    image_size=448,  # 提高图像尺寸以保留更多细节
    flow_steps=16,   # 增加flow步数以提高模型表达能力
    early_stopping_patience=15,
    learning_rate=1e-4,
    output_dir="results"
):
    """
    训练FastFlow异常检测模型
    
    参数:
        data_path: MVTec格式数据集的路径
        category: 数据集类别，默认为"bottle"
        backbone: 特征提取网络, 'resnet18', 'wide_resnet50_2'
        batch_size: 批次大小
        num_workers: 数据加载器的工作进程数
        max_epochs: 最大训练轮数
        image_size: 输入图像尺寸
        flow_steps: Flow模型的步数
        early_stopping_patience: 提前停止的轮数
        learning_rate: 学习率
        output_dir: 输出结果的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard_logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "anomaly_maps"), exist_ok=True)
    
    print("初始化数据模块...")
    
    # 初始化数据模块
    datamodule = MVTecDataModule(
        root=data_path,
        category=category,
        image_size=(image_size, image_size),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task="segmentation"  # 支持异常定位
    )
    
    print("初始化FastFlow模型...")
    
    # 模型配置
    model_config = {
        "backbone": backbone,
        "flow_steps": flow_steps,
        "input_size": image_size,
        "layers": ["layer1", "layer2", "layer3"],  # 使用ResNet的哪些层输出进行异常检测
        "learning_rate": learning_rate
    }
    
    # 初始化增强版FastFlow模型
    model = LitFastflowModel(**model_config)
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="fastflow-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='epoch')  # 添加学习率监控
    ]
    
    # 设置TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "tensorboard_logs"),
        name="fastflow"
    )
    
    # 设置训练器
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # 添加梯度裁剪以提高稳定性
    )
    
    print(f"开始训练FastFlow模型...")
    
    # 训练模型
    trainer.fit(model, datamodule=datamodule)
    
    # 评估模型
    print("开始评估模型...")
    trainer.test(model, datamodule=datamodule)
    
    # 保存训练好的模型配置
    config_path = os.path.join(output_dir, "model_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "model": "FastFlow",
            "backbone": backbone,
            "flow_steps": flow_steps,
            "input_size": image_size,
            "layers": ["layer1", "layer2", "layer3"],
            "checkpoint_path": os.path.join(output_dir, "checkpoints", "last.ckpt")
        }, f)
    
    print(f"模型训练完成！配置已保存到 {config_path}")
    print(f"最佳模型已保存到 {output_dir}/checkpoints/")
    return os.path.join(output_dir, "checkpoints", "last.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastFlow模型训练器")
    parser.add_argument("--data_path", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec数据集类别")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", 
                       choices=["resnet18", "wide_resnet50_2"], help="特征提取器骨干网络")
    parser.add_argument("--batch_size", type=int, default=16, help="训练和评估批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--max_epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--image_size", type=int, default=448, help="输入图像大小")
    parser.add_argument("--flow_steps", type=int, default=16, help="Flow模型的步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        category=args.category,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        image_size=args.image_size,
        flow_steps=args.flow_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    ) 