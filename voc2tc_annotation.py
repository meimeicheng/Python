# -*- coding: utf-8 -*-
"""
將已 Label 的圖檔的 XML 轉換格式
CSV column format is image,id,name,xMin,xMax,yMin,yMax
"""
import os
import argparse
import xml.etree.ElementTree as ET
import csv

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

#    classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
#               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
#               'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    classes = ['word', 'ppt', 'excel'] #office
    fields_header=['image', 'id', 'name', 'xMin', 'xMax', 'yMin', 'yMax']
    
    #img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    #img_inds_file = os.path.join(data_path, 'dataset', data_type + '.txt')
    img_inds_file = os.path.join(data_path, 'dataset', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    #in windows, in order to avoid the newline problem, you must declare it as newline=''
    with open(anno_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields_header)
        for image_ind in image_inds:
            #image_path = os.path.join(data_path, image_ind)
            #annotation = image_path
            #label_path = os.path.join(data_path, os.path.splitext(image_ind)[0] + '.xml')
            #D:\Projects\python\YOLO\data\dataset\VOC\test\VOCdevkit\VOC2007\Annotations
            label_path = os.path.join(data_path, 'images\\app_office_fullscreen\\label', os.path.splitext(os.path.basename(image_ind))[0] + '.xml')
            print(label_path)
            
            #xml檔同時存在，才做轉換
            #if os.path.exists(label_path):
            if os.path.exists(label_path) and os.path.isfile(label_path):
                root = ET.parse(label_path).getroot()
                objects = root.findall('object')
                for obj in objects:
                    difficult = obj.find('difficult').text.strip()
                    if (not use_difficult_bbox) and (int(difficult) == 1):
                        continue
                    bbox = obj.find('bndbox')
                    class_name = obj.find('name').text.lower().strip()
                    class_ind = classes.index(obj.find('name').text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    fields_row=[image_ind, class_ind, class_name, xmin, xmax, ymin, ymax]
                    writer.writerow(fields_row)
                    
#                    print(fields_row)
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="D:\\Projects\\python\\YOLO\\data\\")
    parser.add_argument("--train_annotation", default="D:\\Projects\\python\\YOLO\\data\\dataset\\office_train.csv")
    parser.add_argument("--test_annotation",  default="D:\\Projects\\python\\YOLO\\data\\dataset\\office_test.csv")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(os.path.join(flags.data_path), 'office_train_file', flags.train_annotation, False)
    print('=> The number of image for train is: %d\t' %(num1))
#    print('=> The number of image for train is: %d\tThe number of image for test is:%d' %(num1, num3))


