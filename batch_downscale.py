import os
import cv2
from tqdm import tqdm

def downscale_image(img_path, output_dir, target_width=None, target_height=None, scale=0.5, quality=85):
    """
    降低图片分辨率
    :param img_path: 输入图片路径
    :param output_dir: 输出目录
    :param target_width: 目标宽度（优先级高于scale）
    :param target_height: 目标高度（优先级高于scale）
    :param scale: 缩放比例（当未指定target_width/height时生效）
    :param quality: 输出质量（1-100）
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    # 计算目标尺寸
    h, w = img.shape[:2]
    if target_width and target_height:
        new_size = (target_width, target_height)
    elif target_width:
        new_size = (target_width, int(target_width * h / w))
    elif target_height:
        new_size = (int(target_height * w / h), target_height)
    else:
        new_size = (int(w * scale), int(h * scale))

    # 执行缩放
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # 保存结果
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}{ext}")
    
    # 根据扩展名选择保存参数
    if ext.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext.lower() == '.webp':
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(output_path, resized)  # PNG等无损格式忽略quality参数
    
    return True

def batch_downscale(input_dir, output_dir, **kwargs):
    """
    批量处理目录中的所有图片
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    # 获取图片文件列表
    images = [
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(supported_formats)
    ]

    # 带进度条的批量处理
    success_count = 0
    for img_file in tqdm(images, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        if downscale_image(img_path, output_dir, **kwargs):
            success_count += 1

    print(f"\n处理完成！成功处理 {success_count}/{len(images)} 张图片")
    print(f"输出目录: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量图片降分辨率工具")
    parser.add_argument("input_dir", help="输入图片目录路径")
    parser.add_argument("output_dir", help="输出目录路径")
    parser.add_argument("--width", type=int, help="目标宽度（像素）")
    parser.add_argument("--height", type=int, help="目标高度（像素）")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放比例（0-1之间）")
    parser.add_argument("--quality", type=int, default=85, help="输出质量（1-100）")
    
    args = parser.parse_args()

    # 参数校验
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录不存在 - {args.input_dir}")
        exit(1)

    if args.scale <= 0 or args.scale > 1:
        print("错误：缩放比例必须在0-1之间")
        exit(1)

    # 执行批量处理
    batch_downscale(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_width=args.width,
        target_height=args.height,
        scale=args.scale,
        quality=args.quality
    )
    
# 按比例缩放（缩小到原图的50%）
#python batch_downscale.py input_images output_images --scale 0.5

# 按指定宽度缩放（高度自动按比例计算）
# python batch_downscale.py input_images output_images --width 800

# 同时指定宽高（强制拉伸）
# python batch_downscale.py input_images output_images --width 800 --height 600

# 调整输出质量（仅影响JPEG/WEBP）
# python batch_downscale.py input_images output_images --scale 0.3 --quality 90
