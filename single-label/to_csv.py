""" 
创建训练与验证的csv文件,train和val分别包括两列,第一列为图片地址,第二列为标签
"""
import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET
import random


images_fake = glob.glob('train/*/*')
label_names = os.listdir('train')
label_names.sort()
print(len(label_names))
labels_fake = []
for i in images_fake:
    labels_fake.append(label_names.index(i.split('/')[1]))


def parse_rec(filename):
	"""Parse a PASCAL VOC xml file."""
	tree = ET.parse(filename)
	label =  tree.findall('object')[-1].find('name').text

	return label

images_reality = glob.glob('VOCstatic_base80/JPEGImages/*')
labels_reality = []
for image_path in images_reality:
	xml_path = (image_path.split('.')[0] + '.xml').replace('JPEGImages', 'Annotations')
	true_label = parse_rec(xml_path)
	labels_reality.append(label_names.index(true_label))

reality = list(zip(images_reality, labels_reality))
random.shuffle(reality)
images_reality.clear()
labels_reality.clear()
for i, j in reality:
	images_reality.append(i)
	labels_reality.append(j)
	

split_ratio = 0

data = {'images': images_fake + images_reality, 'labels': labels_fake + labels_reality}
data = pd.DataFrame(data)
train = data[0:int(len(images_fake) + len(images_reality) * split_ratio)]
# train = data[len(images_fake):int(len(images_fake) + len(images_reality) * split_ratio)]
train = train.sample(frac=1).reset_index(drop=True)
# train = train.sample(frac=1).reset_index(drop=True)
val = data[int(len(images_fake) + len(images_reality) * split_ratio):]
print(len(data))
# print(len(train_1))
print(len(train))
print(len(val))
# train_1.to_csv('./csv_0/train_1.csv', index=False, header=False)
train.to_csv('./csv_new/train.csv', index=False, header=False)
val.to_csv('./csv_new/val.csv', index=False, header=False)
