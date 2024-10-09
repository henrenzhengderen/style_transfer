import os
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    加载和处理图像。

    参数:
        image_path: 图片的文件路径

    返回:
        处理后的图像数组，形状为 (1, 256, 256, 3)
    """
    img = Image.open(image_path).convert('RGB')  # 读取图像并转换为RGB模式
    img = img.resize((256, 256), Image.ANTIALIAS)  # 调整图像至256x256
    img_array = np.asarray(img, dtype=np.float32) / 255.0  # 转换为数组并归一化到[0, 1]范围

    # 将像素值归一化至[-1, 1]区间
    img_array = (img_array - 0.5) * 2
    return np.expand_dims(img_array, axis=0)  # 添加batch维度


def load_image_from_folder(folder_path):
    """
    从指定文件夹加载所有图片并返回一个列表。

    参数:
        folder_path: 包含图片的文件夹路径

    返回:
        处理后的图像数组列表
    """
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 过滤文件类型
            image_path = os.path.join(folder_path, filename)
            images.append(load_image(image_path))  # 加载并处理图像

    return np.vstack(images)  # 返回所有图像的数组堆叠


def save_image(img_array, save_path):
    """
    将图像数组保存为文件。

    参数:
        img_array: 图像数组，应该在[-1, 1]范围内
        save_path: 保存的文件路径
    """
    img_array = (img_array + 1) / 2  # 反归一化至[0, 1]
    img_array = np.clip(img_array, 0, 1)  # 确保数值在合法范围内
    img = Image.fromarray((img_array * 255).astype(np.uint8))  # 转换为PIL图像
    img.save(save_path)  # 保存图像文件
