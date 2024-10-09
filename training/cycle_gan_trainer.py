import tensorflow as tf
from tensorflow.keras import layers


# 定义生成器的残差块
def residual_block(x):
    """构建CycleGAN中的残差块"""
    res = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    res = layers.ReLU()(res)
    res = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(res)
    return layers.add([x, res])  # 返回输入和输出的和


# 定义生成器
def build_generator():
    """构建CycleGAN的生成器"""
    inputs = layers.Input(shape=(256, 256, 3))  # 输入层
    x = layers.Conv2D(64, kernel_size=7, strides=1, padding='same')(inputs)  # 初始卷积
    x = layers.ReLU()(x)

    # 下采样
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    # 残差块
    for _ in range(6):
        x = residual_block(x)

    # 上采样
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)  # 输出层
    return tf.keras.Model(inputs, outputs)


# 定义判别器
def build_discriminator():
    """构建CycleGAN的判别器"""
    inputs = layers.Input(shape=(256, 256, 3))  # 输入层

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)  # 初始卷积
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    outputs = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)  # 输出层
    return tf.keras.Model(inputs, outputs)


# 定义对抗损失
def discriminator_loss(real, fake):
    """计算判别器的损失"""
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)  # 真实样本损失
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake), fake)  # 伪造样本损失
    return real_loss + fake_loss  # 总损失


# 定义Cycle一致性损失
def cycle_loss(real_image, cycled_image, lambda_cycle):
    """计算Cycle一致性损失"""
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))  # 绝对差损失
    return lambda_cycle * loss  # 加权的循环损失


# 定义生成器的对抗损失
def generator_loss(fake):
    """计算生成器的损失"""
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)  # 只对生成的图像计算损失


def train_step(generator_g, generator_f, discriminator_x, discriminator_y, image_x, image_y, lambda_cycle):
    """进行一步训练"""

    with tf.GradientTape(persistent=True) as tape:
        # 前向生成
        fake_y = generator_g(image_x, training=True)  # X -> Y
        cycled_x = generator_f(fake_y, training=True)  # Y -> X

        fake_x = generator_f(image_y, training=True)  # Y -> X
        cycled_y = generator_g(fake_x, training=True)  # X -> Y

        # 判别
        disc_real_x = discriminator_x(image_x, training=True)  # 真实X
        disc_real_y = discriminator_y(image_y, training=True)  # 真实Y
        disc_fake_x = discriminator_x(fake_x, training=True)  # 伪造X
        disc_fake_y = discriminator_y(fake_y, training=True)  # 伪造Y

        # 损失计算
        gen_g_loss = generator_loss(disc_fake_y)  # G生成器损失
        gen_f_loss = generator_loss(disc_fake_x)  # F生成器损失

        total_cycle_loss = cycle_loss(image_x, cycled_x, lambda_cycle) + cycle_loss(image_y, cycled_y,
                                                                                    lambda_cycle)  # 循环损失

        # 总生成器损失
        total_gen_g_loss = gen_g_loss + total_cycle_loss
        total_gen_f_loss = gen_f_loss + total_cycle_loss

        # 判别器损失
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # 计算梯度
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # 更新权重
    optimizer_G.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    optimizer_F.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    optimizer_DX.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    optimizer_DY.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss  # 返回损失


def train_model(train_dataset, epochs, lambda_cycle=10):
    """训练CycleGAN模型"""
    # 初始化生成器和判别器
    generator_g = build_generator()
    generator_f = build_generator()
    discriminator_x = build_discriminator()
    discriminator_y = build_discriminator()

    # 使用Adam优化器
    global optimizer_G, optimizer_F, optimizer_DX, optimizer_DY
    optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # 生成器G的优化器
    optimizer_F = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # 生成器F的优化器
    optimizer_DX = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # 判别器X的优化器
    optimizer_DY = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # 判别器Y的优化器

    for epoch in range(epochs):
        for image_x, image_y in train_dataset:
            # 进行一次训练步骤
            gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = train_step(generator_g, generator_f, discriminator_x,
                                                                          discriminator_y, image_x, image_y,
                                                                          lambda_cycle)

        # 在每个epoch结束时输出损失值
        print(
            f'Epoch: {epoch + 1}, Gen G Loss: {gen_g_loss.numpy()}, Gen F Loss: {gen_f_loss.numpy()}, Disc X Loss: {disc_x_loss.numpy()}, Disc Y Loss: {disc_y_loss.numpy()}')
