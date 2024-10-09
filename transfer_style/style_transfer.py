import numpy as np
from PIL import Image
from preprocessing.data_loader import load_image, save_image

def transfer_style(generator, input_image_path, output_image_path):
    """
    使用生成器进行风格迁移。

    参数:
        generator: 已训练好的生成器模型
        input_image_path: 输入图像的文件路径
        output_image_path: 输出图像的保存路径
    """
    # 加载并处理输入图像
    input_image = load_image(input_image_path)  # 调用load_image函数
    # 使用生成器进行风格转换
    generated_image = generator(input_image, training=False)  # 进行图像转换，不使用训练模式

    # 保存生成的图像到指定路径
    save_image(generated_image[0], output_image_path)  # 将生成的图像保存为文件

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

def transfer_style_from_folder(generator, input_folder, output_folder):
    """
    将输入文件夹中的所有图像进行风格迁移并保存到指定输出文件夹。

    参数:
        generator: 已训练好的生成器模型
        input_folder: 输入图像的文件夹路径
        output_folder: 输出图像的文件夹路径
    """
    import os

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 支持的图像类型
            input_image_path = os.path.join(input_folder, filename)  # 输入图像路径
            output_image_path = os.path.join(output_folder, filename)  # 输出图像路径

            # 调用transfer_style函数进行风格迁移
            transfer_style(generator, input_image_path, output_image_path)  # 进行图像转换
            print(f'图像 {filename} 已转换并保存至 {output_image_path}')  # 打印已转换的信息
