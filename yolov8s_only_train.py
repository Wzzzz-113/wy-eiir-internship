from ultralytics import YOLO
import yaml
import os
import numpy as np
import json

def to_int(x):
    return np.round(x, 0).astype(int)

def convert_annotation(input_dir, img_id):
    json_path = os.path.join(input_dir, img_id+'.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            data = json.load(jf)
        H, W = data['imageHeight'], data['imageWidth']

        with open(os.path.join(input_dir, img_id+'.txt'), 'w') as out_file:
            for i in data['shapes']:
                if i['label'] not in labels_dict.values():
                    print('%s not in label list' % i['label'])
                    continue
                if i['shape_type'] == 'rectangle':
                    ps = to_int(i['points'])
                    w_, h_ = ps[1][0] - ps[0][0], ps[1][1] - ps[0][1]
                    xc, yc = ps[0][0] + w_/2, ps[0][1] + h_/2
                    label_index = list(labels_dict.keys())[list(labels_dict.values()).index(i['label'])]
                    out_file.write(f'{label_index} {xc / W} {yc / H} {w_ / W} {h_ / H}\n')
    else:
        print(f"未找到对应的JSON文件: {json_path}")

def generate_image_list_and_labels(input_dir, output_image_list_file):
    image_files = [f for f in os.listdir(input_dir) if f.endswith(suffix) or f.endswith('png')]
    image_files.sort()
    with open(output_image_list_file, 'w') as f:
        for file in image_files:
            f.write(os.path.join(input_dir, file) + '\n')

    for image_file in image_files:
        image_id = image_file.rsplit('.', 1)[0]
        convert_annotation(input_dir, image_id)

def read_labels_txt(file_path):
    """
    读取并解析指定格式的 TXT 文件，将其转换为字典。
    
    参数:
    file_path (str): TXT 文件的路径。
    
    返回:
    dict: 包含标签映射的字典。
    """
    labels = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        for line in lines:
            line = line.strip()
            if line.startswith('names:') or not line:
                continue
        
            key, value = line.split(':', 1)
            key = int(key.strip())
            value = value.strip().strip("'")
            labels[key] = value
    else:
        print(f"未找到标签文件: {file_path}")

    return labels

train_data = '/home/eiir/eiir/pcb_data/pcb_yolo_train_data/train_data/train'
val_data = '/home/eiir/eiir/pcb_data/pcb_yolo_train_data/train_data/val'
labels_dict = read_labels_txt('/home/eiir/eiir/pcb_data/pcb_yolo_train_data/class_names_list.txt')
suffix = 'jpg'

train_data_txt =  train_data + '/train.txt'
val_data_txt = val_data + '/val.txt'
generate_image_list_and_labels(train_data, train_data_txt)
generate_image_list_and_labels(val_data, val_data_txt)


# 加载预训练模型权重
model = YOLO('/home/eiir/eiir/yolo/yolov8s.pt')

with open('new_detect_hyps.yaml') as f:
    hyp_dict = yaml.safe_load(f)

# 训练
imgsz_param = (1280, 800)
model.train(
    data='new_dataset.yaml',       # 你的数据集配置文件，指向包含train/val图像路径和标注格式（通常是yaml）
    epochs=500,                    # 训练轮数，可按需调整
    batch=5,                     # batch大小，按显存调整
    imgsz=imgsz_param,                    # 输入图像大小
    device=0,                     # 使用GPU编号，-1表示CPU
    **hyp_dict,          # 你的超参数yaml文件路径
    pretrained=True,               # 继续用预训练权重做训练
    save=True                   # 是否保存训练结果
)
