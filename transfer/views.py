import os
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from training.cycle_gan_trainer import train_model  # 这里需要根据实际需要来定义train_model
from transfer_style.style_transfer import transfer_style

# 加载和训练模型
# generator_g = load_generator('path/to/your/generator_weights.h5')


def index(request):
    """主页视图"""
    return render(request, 'transfer/index.html')


def train_view(request):
    """训练模型视图"""
    # 假设已经准备好训练数据集
    # train_dataset = prepare_dataset()
    # train_model(train_dataset, epochs=100)  # 示例训练过程
    return HttpResponse("训练结束")


def style_transfer_view(request):
    """风格迁移视图"""
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        output_filename = 'stylized_image.jpg'
        output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

        # 使用生成器进行风格迁移
        # transfer_style(generator_g, os.path.join(settings.MEDIA_ROOT, filename), output_path)

        return render(request, 'transfer/index.html', {
            'uploaded_file_url': uploaded_file_url,
            'output_file_url': f'/media/{output_filename}'
        })
    return render(request, 'transfer/index.html')
