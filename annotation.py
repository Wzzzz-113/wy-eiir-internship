import cv2  
import numpy as np  
import os  
import glob  
from pathlib import Path  
import torch  
from ultralytics import YOLO  

class OBBSemiAutoAnnotator:  
    def __init__(self, image_folder, model_path=None):  
        self.image_folder = image_folder  
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) +   
                                glob.glob(os.path.join(image_folder, "*.png")))  
        self.images = {}  
        self.current_index = 0  
        self.current_class = 0  
        self.points = []  
        self.fitted_box = None  # 存储拟合后的平行四边形  
        self.boxes = []  # 存储格式: [id, class, [x1,y1,x2,y2,x3,y3,x4,y4]]  
        self.box_id = 0  
        self.selected_box = -1  
        self.delete_mode = False  # 删除模式标志  
        
        # YOLO模型相关  
        self.model = None  
        self.model_path = model_path  
        self.auto_detect_enabled = True  # 自动检测开关  
        self.confidence_threshold = 0.3  # 置信度阈值  
        
        # 类别名称和颜色定义  
        self.class_names = {  
            0: "box",  
            1: "soft",   
            2: "envelop",  
            3: "nc",  
            4: "unknown"  
        }  
        
        # YOLO输出类别名称到数字的映射  
        self.yolo_class_mapping = {  
            "box": 0,  
            "soft": 1,  
            "envelop": 2,  
            "envelope": 2,  # 备选名称  
            "nc": 3,  
            "unknown": 4  
        }  
        
        # 为不同类别定义半透明颜色 (B,G,R,A)  
        self.class_colors = {  
            0: (255, 0, 0, 128),    # 红色半透明 - box  
            1: (0, 255, 0, 128),    # 绿色半透明 - soft  
            2: (0, 255, 255, 128),  # 黄色半透明 - envelop  
            3: (0, 0, 255, 128),    # 蓝色半透明 - nc  
            4: (128, 128, 128, 128),# 灰色半透明 - unknown  
        }  
        
        # 加载YOLO模型  
        self.load_yolo_model()  
        
        # 加载所有图片  
        self.load_images()  
        
        # 如果图片文件夹非空，开始显示第一张图片  
        if self.image_paths:  
            self.current_image_path = self.image_paths[self.current_index]  
            self.load_existing_annotations()  
            self.display_image()  
        else:  
            print("文件夹中没有图片!")  
    
    def load_yolo_model(self):  
        """加载YOLO模型"""  
        if self.model_path and os.path.exists(self.model_path):  
            try:  
                self.model = YOLO(self.model_path)  
                print(f"成功加载YOLO模型: {self.model_path}")  
                print(f"模型类别: {self.model.names}")  
                print("自动检测已启用，按'a'键可切换自动检测开关")  
            except Exception as e:  
                print(f"加载YOLO模型失败: {e}")  
                print("将使用纯手动标注模式")  
                self.model = None  
                self.auto_detect_enabled = False  
        else:  
            print("未提供YOLO模型路径或文件不存在，使用纯手动标注模式")  
            self.auto_detect_enabled = False  
    
    def detect_yolo_obbs(self, color_img, target_classes=None, max_nums=20):  
        """  
        基于您的demo修改的YOLO OBB检测函数  
        
        Args:  
            color_img: 输入图像  
            target_classes: 目标类别名称列表  
            max_nums: 最大检测数量  
        
        Returns:  
            obbs: OBB检测结果列表  
        """  
        if self.model is None:  
            return []  
        
        if target_classes is None:  
            target_classes = ['box', 'soft', 'envelop', 'nc']  
        
        # 转换为小写以便匹配  
        target_classes = [cls.lower() for cls in target_classes]  
        
        try:  
            results = self.model(color_img, verbose=False)  
            result = results[0]  
            
            # 检查是否有OBB检测结果  
            if not hasattr(result, 'obb') or result.obb is None:  
                print("模型没有返回OBB检测结果")  
                return []  
            
            # 识别目标类别的ID  
            target_class_ids = {}  
            for i, name in self.model.names.items():  
                if name.lower() in target_classes:  
                    target_class_ids[i] = name.lower()  
            
            if not target_class_ids:  
                print(f"在模型类别中未找到目标类别: {target_classes}")  
                print(f"模型可用类别: {self.model.names}")  
                return []  
            
            # 收集目标类别的OBB检测结果  
            selected_obbs = []  
            
            # 获取OBB检测结果  
            obb_boxes = result.obb.xyxyxyxy  # shape: [N, 4, 2] 或 [N, 8]  
            obb_confs = result.obb.conf      # shape: [N]  
            obb_cls = result.obb.cls         # shape: [N]  
            
            # 转换为CPU numpy数组  
            if hasattr(obb_boxes, "cpu"):  
                obb_boxes = obb_boxes.cpu().numpy()  
            if hasattr(obb_confs, "cpu"):  
                obb_confs = obb_confs.cpu().numpy()  
            if hasattr(obb_cls, "cpu"):  
                obb_cls = obb_cls.cpu().numpy()  
            
            N = len(obb_boxes)  
            for i in range(N):  
                cls_id = int(obb_cls[i])  
                conf = float(obb_confs[i])  
                
                # 置信度过滤  
                if conf < self.confidence_threshold:  
                    continue  
                
                # 检查是否为目标类别  
                if cls_id in target_class_ids:  
                    # 确保OBB坐标格式正确  
                    if len(obb_boxes[i].shape) == 2 and obb_boxes[i].shape == (4, 2):  
                        # 已经是(4,2)格式  
                        obb_points = obb_boxes[i]  
                    elif len(obb_boxes[i]) == 8:  
                        # 是8个坐标的扁平数组，重塑为(4,2)  
                        obb_points = obb_boxes[i].reshape(4, 2)  
                    else:  
                        print(f"未知的OBB格式: {obb_boxes[i].shape}")  
                        continue  
                    
                    selected_obbs.append({  
                        'obb': obb_points,  
                        'conf': conf,  
                        'class': target_class_ids[cls_id],  
                        'class_id': cls_id  
                    })  
            
            # 按置信度排序并限制数量  
            selected_obbs = sorted(selected_obbs, key=lambda b: b['conf'], reverse=True)[:max_nums]  
            
            print(f"检测到 {len(selected_obbs)} 个目标")  
            return selected_obbs  
            
        except Exception as e:  
            print(f"YOLO检测失败: {e}")  
            import traceback  
            traceback.print_exc()  
            return []  
    
    def run_yolo_detection(self):  
        """运行YOLO检测并转换结果"""  
        try:  
            img_name = os.path.basename(self.current_image_path)  
            img = self.images[img_name]  
            
            print(f"正在检测图像: {img_name}")  
            
            # 使用修改后的检测函数  
            detected_obbs = self.detect_yolo_obbs(img)  
            
            h, w = img.shape[:2]  
            detection_count = 0  
            
            for obb_result in detected_obbs:  
                obb_points = obb_result['obb']  # shape: (4, 2)  
                conf = obb_result['conf']  
                class_name = obb_result['class']  
                
                # 映射类别名称到数字ID  
                mapped_class = self.yolo_class_mapping.get(class_name, 4)  # 默认为unknown  
                
                print(f"检测到: {class_name} -> {mapped_class}, 置信度: {conf:.3f}")  
                
                # 转换为归一化坐标 [x1,y1,x2,y2,x3,y3,x4,y4]  
                normalized_coords = []  
                for point in obb_points:  
                    normalized_coords.append(point[0] / w)  # x坐标  
                    normalized_coords.append(point[1] / h)  # y坐标  
                
                # 添加到标注列表  
                self.boxes.append([self.box_id, mapped_class, normalized_coords])  
                self.box_id += 1  
                detection_count += 1  
            
            print(f"YOLO检测完成，找到 {detection_count} 个目标")  
            
        except Exception as e:  
            print(f"YOLO检测失败: {e}")  
            import traceback  
            traceback.print_exc()  
    
    def load_images(self):  
        print(f"正在加载图片，请稍候...")  
        for path in self.image_paths:  
            img_name = os.path.basename(path)  
            self.images[img_name] = cv2.imread(path)  
        print(f"已加载 {len(self.images)} 张图片")  
    
    def load_existing_annotations(self):  
        """加载已存在的标注文件"""  
        self.boxes = []  
        self.box_id = 0  
        
        txt_path = self.current_image_path.rsplit('.', 1)[0] + '.txt'  
        if os.path.exists(txt_path):  
            with open(txt_path, 'r') as f:  
                lines = f.readlines()  
                for line in lines:  
                    values = line.strip().split()  
                    if len(values) == 9:  # class x1 y1 x2 y2 x3 y3 x4 y4  
                        cls = int(values[0])  
                        coords = [float(v) for v in values[1:]]  
                        self.boxes.append([self.box_id, cls, coords])  
                        self.box_id += 1  
        
        # 如果没有现有标注且启用了自动检测，运行YOLO检测  
        if len(self.boxes) == 0 and self.auto_detect_enabled and self.model is not None:  
            self.run_yolo_detection()  
    
    def toggle_auto_detection(self):  
        """切换自动检测开关"""  
        if self.model is not None:  
            self.auto_detect_enabled = not self.auto_detect_enabled  
            status = "启用" if self.auto_detect_enabled else "禁用"  
            print(f"自动检测已{status}")  
        else:  
            print("无可用的YOLO模型")  
    
    def set_confidence_threshold(self, threshold):  
        """设置置信度阈值"""  
        self.confidence_threshold = max(0.0, min(1.0, threshold))  
        print(f"置信度阈值已设置为: {self.confidence_threshold:.2f}")  
    
    def has_annotation(self, image_path):  
        """检查图像是否有标注文件"""  
        txt_path = image_path.rsplit('.', 1)[0] + '.txt'  
        if os.path.exists(txt_path):  
            with open(txt_path, 'r') as f:  
                content = f.read().strip()  
                return len(content) > 0  
        return False  
    
    def find_first_unannotated_image(self):  
        """查找第一个未标注的图像"""  
        for i, path in enumerate(self.image_paths):  
            if not self.has_annotation(path):  
                return i  
        return 0  
    
    def jump_to_unannotated_image(self):  
        """跳转到第一个未标注的图像"""  
        self.save_annotations()  
        
        target_index = self.find_first_unannotated_image()  
        if target_index == self.current_index:  
            print("所有图像都已标注，已定位到第一张图像")  
        else:  
            self.current_index = target_index  
            self.current_image_path = self.image_paths[self.current_index]  
            self.points = []  
            self.fitted_box = None  
            self.selected_box = -1  
            self.delete_mode = False  
            self.load_existing_annotations()  
            print(f"已跳转到第一个未标注的图像: {os.path.basename(self.current_image_path)}")  
        
        self.display_image()  
    
    def get_current_class_color(self, alpha=True):  
        """获取当前类别的颜色"""  
        color = self.class_colors.get(self.current_class, (128, 128, 128, 128))  
        if alpha:  
            return color  
        else:  
            return color[:3]  
    
    def draw_filled_polygon(self, img, points, color):  
        """绘制半透明多边形"""  
        overlay = img.copy()  
        points_array = np.array(points, np.int32)  
        cv2.fillPoly(overlay, [points_array], color[:3])  
        
        alpha = color[3] / 255.0  
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  
    
    def display_image(self):  
        img_name = os.path.basename(self.current_image_path)  
        self.display_img = self.images[img_name].copy()  
        h, w = self.display_img.shape[:2]  
        
        # 显示已标注的平行四边形（包括YOLO检测结果和手动标注）  
        for i, box in enumerate(self.boxes):  
            box_id, cls, coords = box  
            points = []  
            for j in range(0, 8, 2):  
                x = int(coords[j] * w)  
                y = int(coords[j+1] * h)  
                points.append((x, y))  
            
            # 选择颜色  
            color = self.class_colors.get(cls, (128, 128, 128, 128))  
            border_color = (color[0], color[1], color[2])  
            
            # 绘制半透明填充  
            self.draw_filled_polygon(self.display_img, points, color)  
            
            # 绘制边框  
            is_selected = (i == self.selected_box)  
            thickness = 2 if is_selected else 1  
            for j in range(4):  
                cv2.line(self.display_img, points[j], points[(j+1)%4], border_color, thickness)  
            
            # 获取类别名称  
            class_name = self.class_names.get(cls, f"Class {cls}")  
            
            # 显示类别名称和ID  
            label = f"ID:{box_id}, {class_name}"  
            cv2.putText(self.display_img, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  
            cv2.putText(self.display_img, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
        
        # 获取当前类别的颜色用于标注点  
        current_point_color = self.get_current_class_color(alpha=False)  
        
        # 显示当前标注的点（使用当前类别的颜色）  
        for i, point in enumerate(self.points):  
            cv2.circle(self.display_img, point, 5, current_point_color, -1)  
            # 添加白色边框让点更清晰  
            cv2.circle(self.display_img, point, 5, (255, 255, 255), 1)  
            
            # 标记点的序号（使用对比色）  
            text_color = (255, 255, 255) if sum(current_point_color) < 400 else (0, 0, 0)  
            cv2.putText(self.display_img, str(i+1),   
                      (point[0]+8, point[1]-8),   
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)  
        
        # 如果有4个点，显示拟合后的平行四边形  
        if len(self.points) == 4 and self.fitted_box is not None:  
            fitted_points = [(int(x), int(y)) for x, y in self.fitted_box]  
            
            # 半透明填充当前选中的类别颜色  
            color = self.get_current_class_color()  
            self.draw_filled_polygon(self.display_img, fitted_points, color)  
            
            # 用当前类别颜色绘制拟合后的平行四边形边框  
            border_color = self.get_current_class_color(alpha=False)  
            for i in range(4):  
                cv2.line(self.display_img, fitted_points[i], fitted_points[(i+1)%4], border_color, 3)  
                
            # 标记四个角点（使用当前类别颜色，但稍微调亮）  
            bright_color = tuple(min(255, c + 50) for c in border_color)  
            for i, point in enumerate(fitted_points):  
                cv2.circle(self.display_img, point, 6, bright_color, -1)  
                cv2.circle(self.display_img, point, 6, (255, 255, 255), 1)  
                
                # 角点序号  
                text_color = (255, 255, 255) if sum(bright_color) < 400 else (0, 0, 0)  
                cv2.putText(self.display_img, str(i+1),   
                          (point[0]+8, point[1]+8),   
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)  
        
        # 如果有3个点，显示提示  
        if len(self.points) == 3:  
            text = "按Enter键确认三点标注（用于边缘物体）"  
            # 使用当前类别颜色作为提示文字颜色  
            text_color = self.get_current_class_color(alpha=False)  
            cv2.putText(self.display_img, text, (10, h - 50),   
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # 黑色描边  
            cv2.putText(self.display_img, text, (10, h - 50),   
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)  
        
        # 显示当前图片信息和类别（高亮当前类别）  
        delete_str = " | 删除模式" if self.delete_mode else ""  
        auto_str = " | 自动检测:开" if self.auto_detect_enabled else " | 自动检测:关"  
        current_class_name = self.class_names.get(self.current_class, f"Class {self.current_class}")  
        info_text = f"图片: {self.current_index+1}/{len(self.image_paths)} | 当前类别: {self.current_class} ({current_class_name}){delete_str}{auto_str}"  
        cv2.putText(self.display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  
        
        # 显示类别图例（高亮当前选中的类别）  
        legend_start_y = 60  
        for cls_id, name in self.class_names.items():  
            color = self.class_colors.get(cls_id, (128, 128, 128, 128))  
            
            # 如果是当前选中的类别，使用更显眼的显示  
            if cls_id == self.current_class:  
                # 当前类别用更大的矩形和粗边框  
                cv2.rectangle(self.display_img, (8, legend_start_y - 2), (32, legend_start_y + 17), (255, 255, 255), 2)  
                cv2.rectangle(self.display_img, (10, legend_start_y), (30, legend_start_y + 15), color[:3], -1)  
                
                # 当前类别文字用当前类别颜色高亮  
                cv2.putText(self.display_img, f"{cls_id}: {name} (当前)", (38, legend_start_y + 12),   
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  
                cv2.putText(self.display_img, f"{cls_id}: {name} (当前)", (38, legend_start_y + 12),   
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[:3], 1)  
            else:  
                # 其他类别正常显示  
                cv2.rectangle(self.display_img, (10, legend_start_y), (30, legend_start_y + 15), color[:3], -1)  
                cv2.putText(self.display_img, f"{cls_id}: {name}", (35, legend_start_y + 12),   
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  
                cv2.putText(self.display_img, f"{cls_id}: {name}", (35, legend_start_y + 12),   
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
            
            legend_start_y += 25  
        
        # 显示置信度阈值  
        threshold_text = f"置信度阈值: {self.confidence_threshold:.2f}"  
        cv2.putText(self.display_img, threshold_text, (10, legend_start_y + 10),   
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  
        
        # 显示操作指南  
        guide = [  
            "操作指南:",  
            "n: 下一张图片 | b: 上一张图片 | l: 跳转到未标注图像",  
            "0-9: 设置类别 | 左键: 标点/删除 | a: 切换自动检测",  
            "c/d: 删除模式 | u/右键: 撤回点 | Enter: 确认三点标注 | r: 重新检测",  
            "x: 清空标注 | +/-: 调整置信度阈值"  
        ]  
        
        for i, line in enumerate(guide):  
            cv2.putText(self.display_img, line, (10, h - 10 - i*20),   
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  
        
        cv2.imshow("YOLO-OBB Semi-Auto Annotator", self.display_img)  
    
    def fit_parallelogram_from_three_points(self, points):  
        """根据三个点拟合平行四边形，第四个点在图像外"""  
        if len(points) != 3:  
            return None  
        
        h, w = self.display_img.shape[:2]  
        pts = np.array(points, dtype=np.float32)  
        
        # 检查点与图像边界的距离  
        distances_to_boundary = []  
        for pt in pts:  
            dist_left = pt[0]  
            dist_top = pt[1]  
            dist_right = w - pt[0]  
            dist_bottom = h - pt[1]  
            min_dist = min(dist_left, dist_top, dist_right, dist_bottom)  
            distances_to_boundary.append(min_dist)  
        
        # 计算各种可能的第四个点位置  
        possible_fourth_points = []  
        
        for i in range(3):  
            j = (i + 1) % 3  
            k = (i + 2) % 3  
            p4 = pts[k] + (pts[j] - pts[i])  
            
            is_outside = (p4[0] < 0 or p4[0] >= w or p4[1] < 0 or p4[1] >= h)  
            
            if is_outside:  
                ordered_points = self.order_points(np.vstack((pts, [p4])))  
                possible_fourth_points.append((ordered_points, p4))  
        
        if not possible_fourth_points:  
            closest_to_boundary_idx = np.argmin(distances_to_boundary)  
            far_indices = [i for i in range(3) if i != closest_to_boundary_idx]  
            i, j = far_indices  
            
            p4 = pts[closest_to_boundary_idx] + (pts[j] - pts[i])  
            ordered_points = self.order_points(np.vstack((pts, [p4])))  
            return ordered_points  
        
        return possible_fourth_points[0][0]  
    
    def order_points(self, pts):  
        """按顺时针顺序排列四个点"""  
        centroid = np.mean(pts, axis=0)  
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])  
        sorted_indices = np.argsort(angles)  
        return pts[sorted_indices]  
    
    def fit_parallelogram(self, points):  
        """将四个点拟合成平行四边形"""  
        if len(points) != 4:  
            return points  
        
        pts = np.array(points, dtype=np.float32)  
        centroid = np.mean(pts, axis=0)  
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])  
        sorted_indices = np.argsort(angles)  
        pts = pts[sorted_indices]  
        
        p0, p1, p2, p3 = pts  
        
        mid1 = (p0 + p2) / 2  
        mid2 = (p1 + p3) / 2  
        center = (mid1 + mid2) / 2  
        
        v0 = p0 - center  
        v1 = p1 - center  
        v2 = p2 - center  
        v3 = p3 - center  
        
        len0 = np.linalg.norm(v0)  
        len1 = np.linalg.norm(v1)  
        len2 = np.linalg.norm(v2)  
        len3 = np.linalg.norm(v3)  
        
        avg_len02 = (len0 + len2) / 2  
        avg_len13 = (len1 + len3) / 2  
        
        v0_unit = v0 / len0  
        v1_unit = v1 / len1  
        v2_unit = v2 / len2  
        v3_unit = v3 / len3  
        
        new_p0 = center + v0_unit * avg_len02  
        new_p2 = center - v0_unit * avg_len02  
        new_p1 = center + v1_unit * avg_len13  
        new_p3 = center - v1_unit * avg_len13  
        
        parallelogram = np.array([new_p0, new_p1, new_p2, new_p3])  
        return parallelogram  
    
    def add_box_to_annotations(self):  
        """将拟合的平行四边形添加到标注列表中"""  
        if len(self.points) == 4:  
            self.fitted_box = self.fit_parallelogram(self.points)  
            h, w = self.display_img.shape[:2]  
            
            normalized_points = []  
            for point in self.fitted_box:  
                normalized_points.append(point[0] / w)  
                normalized_points.append(point[1] / h)  
            
            self.boxes.append([self.box_id, self.current_class, normalized_points])  
            self.box_id += 1  
            self.points = []  
            self.fitted_box = None  
            self.display_image()  
            return True  
        
        elif len(self.points) == 3:  
            self.fitted_box = self.fit_parallelogram_from_three_points(self.points)  
            if self.fitted_box is not None:  
                h, w = self.display_img.shape[:2]  
                
                normalized_points = []  
                for point in self.fitted_box:  
                    normalized_points.append(point[0] / w)  
                    normalized_points.append(point[1] / h)  
                
                self.boxes.append([self.box_id, self.current_class, normalized_points])  
                self.box_id += 1  
                self.points = []  
                self.fitted_box = None  
                self.display_image()  
                return True  
        
        return False  
    
    def find_box_at_point(self, point):  
        """查找点击位置的框"""  
        h, w = self.display_img.shape[:2]  
        for i, box in enumerate(self.boxes):  
            _, cls, coords = box  
            box_points = []  
            for j in range(0, 8, 2):  
                box_points.append((int(coords[j] * w), int(coords[j+1] * h)))  
            
            if self.point_in_box(point, box_points):  
                return i  
        return -1  
    
    def mouse_callback(self, event, x, y, flags, param):  
        if event == cv2.EVENT_LBUTTONDOWN:  
            if self.delete_mode:  
                box_idx = self.find_box_at_point((x, y))  
                if box_idx >= 0:  
                    removed_box = self.boxes.pop(box_idx)  
                    print(f"已删除标注框 #{removed_box[0]} (类别: {self.class_names.get(removed_box[1], 'unknown')})")  
                else:  
                    print("未选中任何框")  
                self.delete_mode = False  
                self.display_image()  
            else:  
                if len(self.points) < 4:  
                    self.points.append((x, y))  
                    if len(self.points) == 4:  
                        self.fitted_box = self.fit_parallelogram(self.points)  
                        self.add_box_to_annotations()  
                    else:  
                        self.display_image()  
        
        elif event == cv2.EVENT_RBUTTONDOWN:  
            if self.points:  
                self.points.pop()  
                self.fitted_box = None  
                self.display_image()  
    
    def point_in_box(self, point, box_points):  
        """判断点是否在四边形内部"""
        x, y = point
        n = len(box_points)
        inside = False
        
        p1x, p1y = box_points[0]
        for i in range(1, n + 1):
            p2x, p2y = box_points[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def save_annotations(self):
        txt_path = self.current_image_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'w') as f:
            for _, cls, coords in self.boxes:
                line = f"{cls} " + " ".join([f"{c:.6f}" for c in coords]) + "\n"
                f.write(line)
        print(f"标注已保存到 {txt_path}")
    
    def clear_all_annotations(self):
        """清空当前图像的所有标注"""
        self.boxes = []
        self.box_id = 0
        self.points = []
        self.fitted_box = None
        self.selected_box = -1
        print("已清空当前图像的所有标注")
        self.display_image()
    
    def rerun_detection(self):
        """重新运行YOLO检测"""
        if self.model is not None:
            # 清空现有标注
            self.boxes = []
            self.box_id = 0
            self.points = []
            self.fitted_box = None
            # 重新运行检测
            self.run_yolo_detection()
            self.display_image()
            print("已重新运行YOLO检测")
        else:
            print("无可用的YOLO模型")
    
    def adjust_confidence_threshold(self, delta):
        """调整置信度阈值"""
        new_threshold = self.confidence_threshold + delta
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        print(f"置信度阈值调整为: {self.confidence_threshold:.2f}")
        self.display_image()
    
    def run(self):
        cv2.namedWindow("YOLO-OBB Semi-Auto Annotator")
        cv2.setMouseCallback("YOLO-OBB Semi-Auto Annotator", self.mouse_callback)
        
        print("\n=== YOLO-OBB半自动标注程序 ===")
        print("功能说明:")
        print("1. 自动加载YOLO-OBB检测结果作为初始标注")
        print("2. 支持手动添加、删除、修改标注")
        print("3. 类别映射: box->0, soft->1, envelop->2, nc->3")
        print("4. 按'a'切换自动检测开关，按'r'重新检测当前图像")
        print("5. 按'x'清空当前图像所有标注")
        print("6. 按'+'/'-'调整置信度阈值\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC键退出
                self.save_annotations()
                break
            
            elif key == ord('n'):  # 下一张图片
                if self.current_index < len(self.image_paths) - 1:
                    self.save_annotations()
                    
                    self.current_index += 1
                    self.current_image_path = self.image_paths[self.current_index]
                    self.points = []
                    self.fitted_box = None
                    self.selected_box = -1
                    self.delete_mode = False
                    self.load_existing_annotations()
                    self.display_image()
            
            elif key == ord('b'):  # 上一张图片
                if self.current_index > 0:
                    self.save_annotations()
                    
                    self.current_index -= 1
                    self.current_image_path = self.image_paths[self.current_index]
                    self.points = []
                    self.fitted_box = None
                    self.selected_box = -1
                    self.delete_mode = False
                    self.load_existing_annotations()
                    self.display_image()
            
            elif key == ord('l') or key == ord('L'):  # L键跳转到未标注图像
                self.jump_to_unannotated_image()
            
            elif key == ord('a') or key == ord('A'):  # A键切换自动检测
                self.toggle_auto_detection()
                self.display_image()
            
            elif key == ord('r') or key == ord('R'):  # R键重新检测
                self.rerun_detection()
            
            elif key == ord('x') or key == ord('X'):  # X键清空所有标注
                self.clear_all_annotations()
            
            elif ord('0') <= key <= ord('9'):  # 设置类别
                self.current_class = key - ord('0')
                class_name = self.class_names.get(self.current_class, f"Class {self.current_class}")
                print(f"当前类别设置为: {self.current_class} ({class_name})")
                self.display_image()
            
            elif key == 13:  # Enter键，确认框（用于三点标注情况）
                if len(self.points) == 3:
                    self.add_box_to_annotations()
            
            elif key == ord('c') or key == ord('d'):  # 启用删除模式
                self.delete_mode = True
                print("已启用删除模式，点击要删除的框")
                self.display_image()
            
            elif key == ord('u'):  # 撤销最后一个点
                if self.points:
                    self.points.pop()
                    self.fitted_box = None
                    self.display_image()
            
            elif key == ord('s'):  # 保存标注
                self.save_annotations()
            
            elif key == ord('+') or key == ord('='):  # 增加置信度阈值
                self.adjust_confidence_threshold(0.05)
            
            elif key == ord('-') or key == ord('_'):  # 减少置信度阈值
                self.adjust_confidence_threshold(-0.05)
        
        cv2.destroyAllWindows()

def main():
    print("=== YOLO-OBB半自动标注工具 ===")
    
    # 获取图片文件夹路径
    image_folder = input("请输入图片文件夹路径: ")
    if not os.path.exists(image_folder):
        print("图片文件夹不存在!")
        return
    
    # 获取YOLO模型路径（可选）
    model_path = input("请输入YOLO-OBB模型路径 (直接回车跳过，使用纯手动模式): ").strip()
    if not model_path:
        model_path = None
    elif not os.path.exists(model_path):
        print("模型文件不存在，将使用纯手动模式")
        model_path = None
    
    # 创建并运行标注器
    annotator = OBBSemiAutoAnnotator(image_folder, model_path)
    annotator.run()

if __name__ == "__main__":
    main()