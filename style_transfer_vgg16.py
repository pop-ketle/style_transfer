import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras import backend as K

from scipy.optimize import fmin_l_bfgs_b

'''
VGG16を使ったStyle Transferモデル
'''

K.clear_session()

img_width, img_height = 512, 512

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((img_width, img_height))
    img = np.asarray(img, dtype='float32')
    # 平均をゼロにする
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]# RBGからBGRに変換
    img = np.expand_dims(img, axis=0) # (サンプル数, 縦, 横, チャンネル数) の形に変換
    return img

def deprocess_image(x):
    x = x.reshape((img_height, img_width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x

content_image_path = './sample_images/inputs/cat1.jpg'
style_image_path   = './sample_images/inputs/style1.jpg'
output_dir_path    = './sample_images/outputs/'
if not os.path.exists(output_dir_path+'process/'): os.makedirs(output_dir_path+'process/', exist_ok=True)

content_image = preprocess_image(content_image_path)
style_image = preprocess_image(style_image_path)

# Defining variables in keras backend-->tensorflow
content_image = K.variable(content_image)
style_image = K.variable(style_image)
combination_image = K.placeholder((1, img_width, img_height, 3)) # 生成画像の入れ物
print(content_image.shape, style_image.shape, combination_image.shape)

#Concatenate all image data into a single tensor
input_tensor = K.concatenate([content_image, style_image, combination_image], axis = 0)
print(input_tensor.shape)

model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top = False)
# model.summary()
layers = dict([(layer.name, layer.output) for layer in model.layers]) # この書き方かっこいい

# パラメータ
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
iterations = 30

def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# スタイルの損失
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# コンテンツの損失
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 変化に関する損失
def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :])
    b = K.square(x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
    

# コンテンツの損失
loss = K.variable(0.)
layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(content_image_features, combination_features)

# スタイルの損失
feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# 変化に関する損失
loss += total_variation_weight * total_variation_loss(combination_image)

# 損失・勾配を計算
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, img_height, img_width, 3)) - 128.
# x = preprocess_image(content_image_path) # 入力がコンテント画像と同じ方がちょっといい結果が出やすいという情報を得たけどそうでもなかった

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                    fprime=evaluator.grads, maxfun = 20)
    print('Current loss value:', min_val)
    # 生成された画像を保存
    img = deprocess_image(x.copy().reshape((3, img_width, img_height)))
    img = Image.fromarray(img)

    plt.imshow(img)
    img.save(output_dir_path + 'process/{}.jpg'.format(i+1)) # 処理途中の画像
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

img.save(output_dir_path+'out.jpg')