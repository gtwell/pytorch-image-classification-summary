import os
import glob
import xml.etree.ElementTree as ET
import ipdb
import shutil

num2label = {}
label_list = os.listdir('train')
label_list.sort()
for i in label_list:
	num2label[label_list.index(i)] = i

def parse_rec(filename):
	"""Parse a PASCAL VOC xml file."""
	tree = ET.parse(filename)
	global label
	try:
		label =  tree.findall('object')[-1].find('name').text
		drop = ['leshi']
		if label in drop:
			shutil.move(filename, os.path.join('difference', filename))
			img_path = (filename.split('.')[0] + '.jpg').replace('Annotations', 'JPEGImages')
			shutil.move(img_path, os.path.join('difference', img_path))
			print('moving is going on ...')
	except:
		print(filename)
	# label =  tree.find('folder').text

	return label

def tongji_voc80():
    images = glob.glob('VOCstatic_base80/JPEGImages/*')
    true_labels = {}
    labels_set = set()

    for image_path in images:
	    xml_path = (image_path.split('.')[0] + '.xml').replace('JPEGImages', 'Annotations')
	    true_label = parse_rec(xml_path)
	    true_labels[true_label] = 1 + true_labels.get(true_label, 0)
	    labels_set.add(true_label)
		# true_labels.add(true_label)
	# with open('data.txt', 'w') as f:	
	# 	for i, j in true_labels.items():
	# 		f.write(i+': '+str(j)+'\n')
	# new_label = labels_set.difference(set(label_list))
	# new_label_1 = set(label_list).difference(labels_set)
	# print(len(labels_set))
    print(len(labels_set))
    print(len(label_list))


tongji_voc80()
