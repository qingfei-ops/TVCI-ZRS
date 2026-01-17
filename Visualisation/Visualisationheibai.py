import json
import cv2
import numpy as np
import os
from pathlib import Path
import random


class BWMaskComparisonVisualizer:
    def __init__(self, results_json, annotation_json, image_dir, output_dir='visualization_output'):
        """
        初始化可视化器 - 左右对比显示原图和黑白分割掩码

        Args:
            results_json: COCO格式的检测结果JSON文件路径
            annotation_json: COCO格式的标注JSON文件路径
            image_dir: 原始图像目录
            output_dir: 可视化结果输出目录
        """
        self.results_json = results_json
        self.annotation_json = annotation_json
        self.image_dir = image_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载检测结果
        print(f"加载检测结果: {results_json}")
        with open(results_json, 'r') as f:
            self.results = json.load(f)
        print(f"共 {len(self.results)} 个检测结果")

        # 加载标注文件获取图像信息
        print(f"加载标注文件: {annotation_json}")
        with open(annotation_json, 'r') as f:
            self.annotations = json.load(f)

        # 创建image_id到文件名的映射
        self.image_id_to_filename = {}
        for img in self.annotations['images']:
            self.image_id_to_filename[img['id']] = img['file_name']

        print(f"共 {len(self.image_id_to_filename)} 张图像")

        # 按image_id组织结果
        self.results_by_image = {}
        for result in self.results:
            img_id = result['image_id']
            if img_id not in self.results_by_image:
                self.results_by_image[img_id] = []
            self.results_by_image[img_id].append(result)

    def _decode_mask(self, segmentation, img_shape):
        """解码RLE格式的分割掩码"""
        try:
            # 如果是RLE格式
            if isinstance(segmentation, dict) and 'counts' in segmentation:
                from pycocotools import mask as mask_utils
                if isinstance(segmentation['counts'], list):
                    rle = mask_utils.frPyObjects(segmentation, img_shape[0], img_shape[1])
                else:
                    rle = segmentation
                mask = mask_utils.decode(rle)
                return mask
            # 如果是polygon格式
            elif isinstance(segmentation, list):
                mask = np.zeros(img_shape, dtype=np.uint8)
                for poly in segmentation:
                    poly = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
                return mask
        except Exception as e:
            print(f"    解码掩码失败: {e}")
            return None

    def visualize_image(self, image_id, score_threshold=0.3, add_text=True):
        """
        可视化单张图像 - 左右对比原图和黑白分割掩码

        Args:
            image_id: 图像ID
            score_threshold: 置信度阈值
            add_text: 是否添加标题文字
        """
        # 获取文件名
        if image_id not in self.image_id_to_filename:
            print(f"错误: 图像ID {image_id} 不在标注文件中")
            return None

        filename = self.image_id_to_filename[image_id]

        # 读取图像
        img_path = os.path.join(self.image_dir, filename)
        if not os.path.exists(img_path):
            print(f"错误: 图像不存在: {img_path}")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法读取图像: {img_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 创建黑白掩码图像（黑色背景）
        h, w = img.shape[:2]
        mask_img = np.zeros((h, w, 3), dtype=np.uint8)  # 黑色背景

        # 获取该图像的所有检测结果
        if image_id not in self.results_by_image:
            print(f"警告: 图像 {image_id} ({filename}) 没有检测结果")
        else:
            detections = self.results_by_image[image_id]

            # 过滤低置信度的检测
            detections = [d for d in detections if d['score'] >= score_threshold]

            if len(detections) == 0:
                print(f"图像 {filename} 没有超过阈值 {score_threshold} 的检测")
            else:
                print(f"图像 {filename} 有 {len(detections)} 个分割掩码")

                # 绘制每个实例的掩码（白色）
                for idx, det in enumerate(detections):
                    if 'segmentation' not in det:
                        continue

                    # 解码掩码
                    mask = self._decode_mask(det['segmentation'], img.shape[:2])
                    if mask is None:
                        continue

                    # 将掩码区域设置为白色
                    mask_img[mask > 0] = [255, 255, 255]

        # 创建左右对比图
        if add_text:
            text_height = 50
            comparison = np.ones((h + text_height, w * 2, 3), dtype=np.uint8) * 255

            # 放置原图和黑白掩码图
            comparison[text_height:, :w] = img
            comparison[text_height:, w:] = mask_img

            # 添加标题文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            # 原图标题
            text1 = "Original Image"
            text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
            text_x1 = (w - text_size1[0]) // 2
            cv2.putText(comparison, text1, (text_x1, 35),
                        font, font_scale, (0, 0, 0), thickness)

            # 分割图标题
            text2 = "Binary Mask"
            text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
            text_x2 = w + (w - text_size2[0]) // 2
            cv2.putText(comparison, text2, (text_x2, 35),
                        font, font_scale, (0, 0, 0), thickness)

            # 添加分隔线
            cv2.line(comparison, (w, text_height), (w, h + text_height), (200, 200, 200), 2)
        else:
            # 不添加标题，直接拼接
            comparison = np.hstack([img, mask_img])

        # 保存结果
        output_filename = f'bw_comparison_{filename}'
        output_path = os.path.join(self.output_dir, output_filename)
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, comparison_bgr)
        print(f"  已保存到: {output_path}")

        return comparison

    def visualize_all(self, max_images=50, score_threshold=0.3, add_text=True):
        """
        批量可视化多张图像的对比图

        Args:
            max_images: 最多处理的图像数量
            score_threshold: 置信度阈值
            add_text: 是否添加标题文字
        """
        # 获取所有有检测结果的图像ID
        image_ids = list(self.results_by_image.keys())[:max_images]

        print(f"\n开始可视化 {len(image_ids)} 张图像的黑白掩码对比...")
        print("=" * 60)

        success_count = 0
        for idx, img_id in enumerate(image_ids):
            print(f"\n[{idx + 1}/{len(image_ids)}] 处理图像ID: {img_id}")
            try:
                result = self.visualize_image(img_id, score_threshold=score_threshold, add_text=add_text)
                if result is not None:
                    success_count += 1
            except Exception as e:
                print(f"  处理失败: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print(f"完成! 成功处理 {success_count}/{len(image_ids)} 张图像")
        print(f"结果保存在: {self.output_dir}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    results_json = ""
    annotation_json = ""
    image_dir = ""
    output_dir = ""

    # 创建可视化器
    print("初始化黑白掩码对比可视化器...")
    visualizer = BWMaskComparisonVisualizer(
        results_json=results_json,
        annotation_json=annotation_json,
        image_dir=image_dir,
        output_dir=output_dir
    )

    # 可视化所有图像的左右对比
    print("\n开始可视化...")
    visualizer.visualize_all(
        max_images=100,  # 可视化前100张
        score_threshold=0.1,  # 置信度阈值
        add_text=True  # 添加标题文字
    )