import os
import numpy as np
import codecs
import cv2


class Array2voc:
    def __init__(self,
                 saved_path="./VOCdevkit/VOC2007/"):
        self.saved_path = saved_path  # 保存路径
        self.image_save_path = "ImageSets/Main/"
        self.image_raw_path = "./RawImages/"
        if not os.path.exists(self.saved_path + "Annotations"):
            os.makedirs(self.saved_path + "Annotations")
        if not os.path.exists(self.saved_path + "JPEGImages/"):
            os.makedirs(self.saved_path + "JPEGImages/")
        if not os.path.exists(self.saved_path + "ImageSets/Main/"):
            os.makedirs(self.saved_path + "ImageSets/Main/")

    def save_one_img(self, img, img_name, labels):
        # img_path = 'name0.jpg'
        # labels = np.array([[1, 2, 12, 14, 'label0'],
        #                    [1, 2, 12, 14, 'label1'],
        #                    ])
        total_csv_annotations = dict()
        total_csv_annotations[img_name] = labels
        for filename, label in total_csv_annotations.items():
            assert img is not None
            height, width, channels = img.shape
            with codecs.open(self.saved_path + "Annotations/" + filename.replace(".jpg", ".xml"), "w", "utf-8") as xml:
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
                xml.write('\t<filename>' + filename + '</filename>\n')
                xml.write('\t<source>\n')
                xml.write('\t\t<database>The UAV autolanding</database>\n')
                xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
                xml.write('\t\t<image>flickr</image>\n')
                xml.write('\t\t<flickrid>NULL</flickrid>\n')
                xml.write('\t</source>\n')
                xml.write('\t<owner>\n')
                xml.write('\t\t<flickrid>NULL</flickrid>\n')
                xml.write('\t\t<name>YongLe</name>\n')
                xml.write('\t</owner>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
                if isinstance(label, float):
                    xml.write('</annotation>')
                    continue
                for label_detail in label:
                    labels = label_detail
                    xmin = int(labels[0])
                    ymin = int(labels[1])
                    xmax = int(labels[2])
                    ymax = int(labels[3])
                    label_ = labels[-1]
                    if xmax <= xmin:
                        pass
                    elif ymax <= ymin:
                        pass
                    else:
                        xml.write('\t<object>\n')
                        xml.write('\t\t<name>' + label_ + '</name>\n')
                        xml.write('\t\t<pose>Unspecified</pose>\n')
                        xml.write('\t\t<truncated>1</truncated>\n')
                        xml.write('\t\t<difficult>0</difficult>\n')
                        xml.write('\t\t<bndbox>\n')
                        xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                        xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                        xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                        xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                        xml.write('\t\t</bndbox>\n')
                        xml.write('\t</object>\n')

                xml.write('</annotation>')

        img_save_path = self.saved_path + self.image_save_path + img_name
        print("img_save_path:", img_save_path)
        cv2.imwrite(img_save_path, img)


def main():
    array2voc = Array2voc()
    for i in range(1000):
        img_name = "name"+str(i)+".jpg"
        img_path = "./RawImages/"
        labels = np.array([[1, 2, 12, 14, 'label0'],
                           [1, 2, 12, 14, 'label1'],
                           ])
        # path = img_path+img_name
        path = img_path + "name0.jpg"
        img = cv2.imread(path)
        array2voc.save_one_img(img=img,
                               labels=labels,
                               img_name=img_name)


if __name__=="__main__":
    main()
