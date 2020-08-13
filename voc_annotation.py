import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'),
      ('2007', 'val'),
      ('2007', 'test')]

# 修改类比
classes = ["cube"]


def convert_annotation(year, image_id, list_file):
    str_image_index = image_id.find('image')
    xml_id = image_id[str_image_index:-4]
    in_file = open('/home/lyl/yolo_data/VOC_small_csv/VOC%s/Annotations/%s.xml' % (year, xml_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()


for year, image_set in sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    image_ids = open('/home/lyl/yolo_data/VOC_small_csv/VOC%s/ImageSets/Main/%s.txt' %
                     (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        try:
            str_image_index = image_id.find('image')
            xml_id = image_id[str_image_index:-4]
            list_file.write('/home/lyl/yolo_data/VOC_small_csv/VOC2007/ImageSets/Main/%s.jpg'%(xml_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        except Exception as e:
            print("e:", e)
            print("image_id:", image_id[107:-4])
    list_file.close()
