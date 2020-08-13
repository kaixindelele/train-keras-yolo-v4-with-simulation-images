import os
import random 

# 修改一下标签路径和存储路径
xmlfilepath=r'/home/lyl/yolo_data/VOC_small_csv/VOC2007/Annotations'
saveBasePath=r"/home/lyl/yolo_data/VOC_small_csv/VOC2007/ImageSets/Main/"
 
trainval_percent = 1
train_percent = 1

temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  


# cwd_path = os.getcwd()
for i in list:  
    # total_xml[i] -- image1_2.xml  
    # 修改一下需要保存路径
    name = "/home/lyl/yolo_data/VOC_small_csv/VOC2007/ImageSets/Main/" +total_xml[i][:-4] + '.jpg\n'

    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
