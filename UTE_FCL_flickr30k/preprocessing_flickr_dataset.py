import pandas as pd
import numpy as np
import os
import shutil
import pickle


# 读取数据
annotations = pd.read_table('./flickr30k/results_20130124.token', sep='\t', header=None,
                             names=['image', 'caption'])

# 获取每张图片对应的caption
annotations['image_id'] = annotations['image'].str.split('#').str[0]
grouped = annotations.groupby('image_id')

# 获取所有图片ID和对应的captions
image_data = []
for img_id, group in grouped:
    captions = group['caption'].values
    image_data.append((img_id, captions))


# 打乱顺序
np.random.shuffle(image_data)

# 划分数据集
num_images = len(image_data)
train_data = image_data[:int(num_images * 0.8)]
val_data = image_data[int(num_images * 0.8):int(num_images * 0.9)]
test_data = image_data[int(num_images * 0.9):]

# 准备数据格式
def create_dataset(image_data):
    dataset = {}
    ids = []
    for img_id, captions in image_data:
        entry = {
            'title': captions[0],  # #0的caption
            'ingredients': captions[1:3].tolist(),  # #1和#2的caption
            'instructions': captions[3:5].tolist(),  # #3和#4的caption
            'images': [img_id]  # 唯一的图像名称
        }
        dataset[img_id] = entry
        # print("img_id", img_id, "entry", entry)
        ids.append(img_id)
    return dataset, ids

# 创建训练集、验证集和测试集
train_data, train_ids = create_dataset(train_data)
val_data, val_ids = create_dataset(val_data)
test_data, test_ids = create_dataset(test_data)



# 保存为pkl文件
with open('train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('val.pkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('test.pkl', 'wb') as f:
    pickle.dump(test_data, f)


# 创建目标文件夹
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)


# 定义移动图片的函数
def move_images(ids, folder):
    for img_id in ids:
        img_name = img_id  # 图片名称
        src_path = os.path.join('flickr30k-images', img_name)  # 原路径
        dest_path = os.path.join(folder, img_name)  # 目标路径

        try:
            shutil.copy(src_path, dest_path)  # 复制图片
        except Exception as e:
            print(f"Error moving {img_name}: {e}")


# 移动训练集、验证集和测试集的图片
move_images(train_ids, 'train')
move_images(val_ids, 'val')
move_images(test_ids, 'test')


