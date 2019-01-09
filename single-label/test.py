from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from models import create_model
import os
import glob
import xml.etree.ElementTree as ET
import ipdb


num2label = {}
label_list = os.listdir('train')
label_list.sort()
for i in label_list:
	num2label[label_list.index(i)] = i

imsize = 224
loader = transforms.Compose([
		transforms.Resize((imsize, imsize)), 
		transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    # image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU

model_conv = create_model(model_key='resnet18',
			pretrained='False',
			num_of_classes=75,
			device=device)

model_conv.load_state_dict(torch.load('saved_model_file/csv/resnet18/resnet18_Adam.pkl'))
model_conv.eval()

def parse_rec(filename):
	"""Parse a PASCAL VOC xml file."""
	tree = ET.parse(filename)
	# label_ =  tree.findall('object')[-1].find('name').text
	label =  tree.find('folder').text

	return label


def test_fake():
	i = 0
	images = glob.glob('train/wanglaoji/*')
	for img in images:
		image = image_loader(img)
		true_label = img.split('/')[1]
		output = model_conv(image)
		output = torch.nn.functional.softmax(output, dim=1)
		label = torch.argmax(output).item()
		if true_label == num2label[label]:
			print('true_label: {}'.format(true_label))
			print('pre_label: {} \n'.format(num2label[label]))
			i += 1
			if i % 50 == 0:
				print('right: {}'.format(i))

	print('finished, images predicted rightly is: {}'.format(i))


def test_voc80():
	i = 0
	images = glob.glob('VOCstatic_base80/JPEGImages/*')
	# true_labels = set()

	for image_path in images:
		image = image_loader(image_path)
		# print(image_path)
		xml_path = (image_path.split('.')[0] + '.xml').replace('JPEGImages', 'Annotations')
		true_label = parse_rec(xml_path)
		# true_labels.add(true_label)
		output = model_conv(image)
		output = torch.nn.functional.softmax(output, dim=1)
		label = torch.argmax(output).item()
		if true_label == num2label[label]:
			print('true_label: {}'.format(true_label))
			print('pre_label: {} \n'.format(num2label[label]))
			i += 1
			if i % 50 == 0:
				print('right: {}'.format(i))

	print('finished, images predicted rightly is: {}'.format(i))
	#print(true_labels.difference(set(label_list)))

test_voc80()
