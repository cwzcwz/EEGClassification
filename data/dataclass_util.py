"""preprocess dataset"""

import os
import random

train_percent = 0.9  # 训练数据和交叉验证数据占的比例，自己根据实际调节
val_percent = 0.1  # 训练数据占trainval的比例，即用来训练的数据
allfilepath = './all'
satisdaction_dataset = './satisfaction'
dissatisfaction_dataset = './dissatisfaction'
txtsavepath = './'
total_file = os.listdir(allfilepath)

num = len(total_file)
list = range(num)
tv = int(num * train_percent)
train = random.sample(list, tv)

ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

for i in list:
    name = total_file[i][:-4] + '\n'
    image_name = total_file[i][:-4]
    for indoor_name in os.listdir(satisdaction_dataset):
        if indoor_name[:-4] == image_name:
            name = image_name + ' 0' + '\n'
    for outdoor_name in os.listdir(dissatisfaction_dataset):
        if outdoor_name[:-4] == image_name:
            name = image_name + ' 1' + '\n'

    if i in train:
        ftrain.write(name)
    else:
        fval.write(name)

ftrain.close()
fval.close()
