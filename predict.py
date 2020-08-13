from yolo import YOLO
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# yolo = YOLO()
# model_path = 'model_data/yolo4_keras_weight.h5'
model_path = '/home/lyl/Documents/yolov4-keras-master/nb_logs/ep006-loss12.358-val_loss6.002.h5'
yolo = YOLO(model_path=model_path)


# img_file = '/home/lyl/Documents/yolov4-tf2-master/robosuite_save_data/connect_dataset/VOCdevkit/VOC2007/ImageSets/Main/'
img_file = 'img'
i = 0
for root, dirs, files in os.walk(img_file):
    for f in files:
        img_path = os.path.join(root, f)
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image_array = np.array(r_image).astype('uint8')
            plt.imshow(r_image_array)
            plt.pause(0.5)
            # plt.pause(1.0)
            # if i == 0:
            #     import time
            #     time.sleep(3)
            # r_image.show()

        i += 1

        # if i > 10:
        #     break
