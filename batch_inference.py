import os
import argparse
import csv
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from inference import load_model, preprocess_image, detect_anomaly, detect_text_regions, dynamic_threshold

def visualize_batch_results(original_image, anomaly_map, anomaly_score, output_path=None, threshold_ratio=None):
    """
    可视化批量推理的异常检测结果，仅用矩形框标注缺陷区域，重点关注文字区域
    
    参数:
        original_image: 原始图像
        anomaly_map: 异常热力图
        anomaly_score: 异常分数
        output_path: 结果保存路径
        threshold_ratio: 阈值比例，用于二值化异常图，如果为None则使用动态阈值
    """
    # 检测文字区域
    text_mask = detect_text_regions(original_image)
    
    # 动态确定阈值
    if threshold_ratio is None:
        threshold_ratio = dynamic_threshold(anomaly_score)
    
    # 将异常图调整为与原始图像相同尺寸
    h, w = original_image.shape[:2]
    anomaly_map_resized = cv2.resize(anomaly_map[0, 0], (w, h))
    
    # 归一化异常图到0-1，然后缩放到0-255
    anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
    anomaly_map_norm_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
    
    # 计算阈值，大于阈值的认为是异常区域
    threshold_value = threshold_ratio * anomaly_map_norm.max()
    _, binary_map = cv2.threshold(anomaly_map_norm_uint8, int(threshold_value * 255), 255, cv2.THRESH_BINARY)
    
    # 将文字区域掩码应用到二值图上，重点检测文字区域的异常
    binary_map = cv2.bitwise_and(binary_map, text_mask)
    
    # 形态学操作，连接相邻异常区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
    
    # 寻找轮廓并在原图上标记
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anomaly_image = original_image.copy()
    
    # 绘制轮廓，将异常区域用红色矩形框标注
    min_contour_area = 10  # 降低最小轮廓面积以检测更小的缺陷
    has_anomaly = False
    defect_count = 0
    defect_areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            has_anomaly = True
            defect_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            defect_areas.append(area)
            cv2.rectangle(anomaly_image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 红色框
    
    # 创建可视化图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(original_image)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    title = f"检测结果 (分数: {anomaly_score:.4f}, 阈值: {threshold_ratio:.2f})"
    if defect_count > 0:
        title += f", 检测到{defect_count}处缺陷"
    plt.title(title)
    plt.imshow(anomaly_image)
    
    # 如果有缺陷，在图上标注"异常"，否则标注"正常"
    if has_anomaly and anomaly_score > 0.5:
        status = "异常"
        color = 'red'
    else:
        status = "正常"
        color = 'green'
    
    plt.text(10, 30, status, fontsize=16, color=color, weight='bold', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    plt.close()
    
    # 返回异常区域信息
    anomaly_info = {
        "has_anomaly": has_anomaly and anomaly_score > 0.5,
        "anomaly_score": anomaly_score,
        "defect_count": defect_count,
        "defect_areas": defect_areas if defect_areas else [],
        "threshold_used": threshold_ratio
    }
    
    return anomaly_image, anomaly_info

def process_directory(
    image_dir, 
    model_config_path, 
    output_dir="results/batch_predictions", 
    threshold_ratio=None,
    anomaly_threshold=0.5,
    save_images=True
):
    """
    批量处理目录中的图像
    
    参数:
        image_dir: 图像目录路径
        model_config_path: 模型配置文件路径
        output_dir: 输出目录
        threshold_ratio: 异常区域阈值比例，None表示使用动态阈值
        anomaly_threshold: 异常分数阈值
        save_images: 是否保存可视化结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("加载模型...")
    model, device, input_size = load_model(model_config_path)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(Path(image_dir).glob(ext)))
    
    if not image_files:
        print(f"在目录 '{image_dir}' 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    # 创建CSV文件记录结果
    csv_path = os.path.join(output_dir, "anomaly_detection_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['图像名称', '异常分数', '检测结果', '缺陷数量', '平均缺陷面积', '使用阈值']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 创建子目录用于分类
        normal_dir = os.path.join(output_dir, "normal")
        anomaly_dir = os.path.join(output_dir, "anomaly")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)
        
        # 统计信息
        anomaly_count = 0
        defect_count = 0
        
        # 批量处理图像
        for img_path in tqdm(image_files):
            try:
                # 预处理图像
                image_tensor, original_image = preprocess_image(str(img_path), input_size)
                
                # 检测异常
                anomaly_map, anomaly_score = detect_anomaly(model, image_tensor, device)
                
                # 可视化并保存结果
                if save_images:
                    save_dir = anomaly_dir if anomaly_score > anomaly_threshold else normal_dir
                    output_path = os.path.join(save_dir, f"{img_path.stem}_result.png")
                    
                    # 可视化不显示，只保存
                    _, anomaly_info = visualize_batch_results(
                        original_image, anomaly_map, anomaly_score, output_path, threshold_ratio
                    )
                else:
                    # 只计算异常信息，不保存图像
                    _, anomaly_info = visualize_batch_results(
                        original_image, anomaly_map, anomaly_score, None, threshold_ratio
                    )
                
                # 判断是否为异常
                is_anomaly = anomaly_info["has_anomaly"]
                if is_anomaly:
                    anomaly_count += 1
                
                defect_count += anomaly_info["defect_count"]
                
                # 计算平均缺陷面积
                avg_area = 0
                if anomaly_info["defect_count"] > 0:
                    avg_area = sum(anomaly_info["defect_areas"]) / len(anomaly_info["defect_areas"])
                
                # 记录结果
                image_name = img_path.name
                writer.writerow({
                    '图像名称': image_name,
                    '异常分数': f"{anomaly_score:.4f}",
                    '检测结果': '异常' if is_anomaly else '正常',
                    '缺陷数量': anomaly_info["defect_count"],
                    '平均缺陷面积': f"{avg_area:.1f}",
                    '使用阈值': f"{anomaly_info['threshold_used']:.2f}"
                })
            
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {str(e)}")
    
    # 输出汇总统计
    print("\n处理完成!")
    print(f"处理图像总数: {len(image_files)}")
    print(f"检测到的异常图像数: {anomaly_count} ({anomaly_count/len(image_files)*100:.1f}%)")
    print(f"检测到的缺陷总数: {defect_count}")
    print(f"结果已保存到: {csv_path}")
    
    # 根据异常分数对图像进行排序的结果
    print("\n异常分数最高的5张图像:")
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        sorted_results = sorted(reader, key=lambda x: float(x['异常分数']), reverse=True)
        
        for i, row in enumerate(sorted_results[:5]):
            print(f"{i+1}. {row['图像名称']} - 分数: {row['异常分数']} - 缺陷数量: {row['缺陷数量']}")
    
    return csv_path

def generate_summary_report(csv_path, output_dir):
    """
    生成检测结果的汇总报告
    
    参数:
        csv_path: 检测结果CSV文件路径
        output_dir: 输出目录
    """
    # 读取CSV结果
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        results = list(reader)
    
    # 提取统计数据
    total_images = len(results)
    anomaly_results = [r for r in results if r['检测结果'] == '异常']
    anomaly_count = len(anomaly_results)
    normal_count = total_images - anomaly_count
    
    # 计算缺陷总数
    total_defects = sum(int(r['缺陷数量']) for r in results)
    
    # 按异常分数排序
    sorted_results = sorted(results, key=lambda x: float(x['异常分数']), reverse=True)
    
    # 计算异常分数分布
    scores = [float(r['异常分数']) for r in results]
    
    # 创建汇总图表
    plt.figure(figsize=(15, 12))
    
    # 1. 饼图：正常/异常比例
    plt.subplot(2, 2, 1)
    plt.pie([normal_count, anomaly_count], 
            labels=['正常', '异常'], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            startangle=90)
    plt.title('检测结果分布')
    
    # 2. 柱状图：异常分数分布
    plt.subplot(2, 2, 2)
    plt.hist(scores, bins=20, color='#2196F3', alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='异常阈值')
    plt.xlabel('异常分数')
    plt.ylabel('图像数量')
    plt.title(f'异常分数分布 (总缺陷数: {total_defects})')
    plt.legend()
    
    # 3. 缺陷数量分布
    defect_counts = [int(r['缺陷数量']) for r in results]
    unique_counts = sorted(list(set(defect_counts)))
    count_freq = [defect_counts.count(c) for c in unique_counts]
    
    plt.subplot(2, 2, 3)
    plt.bar(range(len(unique_counts)), count_freq, tick_label=unique_counts)
    plt.xlabel('缺陷数量')
    plt.ylabel('图像数量')
    plt.title('缺陷数量分布')
    
    # 4. 表格：异常分数最高的10张图像
    plt.subplot(2, 1, 2)
    table_data = [[i+1, r['图像名称'], r['异常分数'], r['检测结果'], r['缺陷数量'], r['平均缺陷面积']] 
                  for i, r in enumerate(sorted_results[:10])]
    plt.table(cellText=table_data,
              colLabels=['序号', '图像名称', '异常分数', '检测结果', '缺陷数量', '平均缺陷面积'],
              loc='center',
              cellLoc='center',
              colWidths=[0.05, 0.3, 0.15, 0.15, 0.15, 0.15])
    plt.axis('off')
    plt.title('异常分数最高的10张图像')
    
    # 保存报告
    report_path = os.path.join(output_dir, "anomaly_detection_report.png")
    plt.tight_layout()
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"汇总报告已保存到: {report_path}")
    
    # 额外生成详细文本报告
    text_report_path = os.path.join(output_dir, "anomaly_detection_report.txt")
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("=============== 瓶子印刷质量异常检测报告 ===============\n\n")
        f.write(f"总检测图像数: {total_images}\n")
        f.write(f"异常图像数: {anomaly_count} ({anomaly_count/total_images*100:.1f}%)\n")
        f.write(f"正常图像数: {normal_count} ({normal_count/total_images*100:.1f}%)\n")
        f.write(f"检测到的缺陷总数: {total_defects}\n")
        f.write(f"平均每张异常图像的缺陷数: {total_defects/max(1, anomaly_count):.2f}\n\n")
        
        f.write("异常类型分布:\n")
        f.write(f"- 有缺陷图像数: {len([r for r in results if int(r['缺陷数量']) > 0])}\n")
        
        f.write("\n异常分数最高的10张图像:\n")
        for i, r in enumerate(sorted_results[:10]):
            f.write(f"{i+1}. {r['图像名称']} - 分数: {r['异常分数']} - 缺陷数量: {r['缺陷数量']} - 平均面积: {r['平均缺陷面积']}\n")
    
    print(f"详细文本报告已保存到: {text_report_path}")
    
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastFlow批量异常检测")
    parser.add_argument("--image_dir", type=str, required=True, help="包含测试图像的目录")
    parser.add_argument("--config_path", type=str, default="results/model_config.yaml", help="模型配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results/batch_predictions", help="结果输出目录")
    parser.add_argument("--threshold_ratio", type=float, default=None, 
                       help="异常区域阈值比例，不指定则使用动态阈值")
    parser.add_argument("--anomaly_threshold", type=float, default=0.5, help="异常分数阈值")
    parser.add_argument("--save_images", action="store_true", help="是否保存所有可视化结果")
    
    args = parser.parse_args()
    
    # 处理图像目录
    csv_path = process_directory(
        image_dir=args.image_dir,
        model_config_path=args.config_path,
        output_dir=args.output_dir,
        threshold_ratio=args.threshold_ratio,
        anomaly_threshold=args.anomaly_threshold,
        save_images=args.save_images
    )
    
    # 生成汇总报告
    if csv_path:
        generate_summary_report(csv_path, args.output_dir) 