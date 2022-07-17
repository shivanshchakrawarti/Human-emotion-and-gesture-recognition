import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

data = pd.read_csv('fer20131.csv')
data.tail()

data.drop(data[data['emotion'] == 1].index, inplace = True)

labels = []
usage = []

for i in data["emotion"]:
    labels.append(i)
    
for i in data["Usage"]:
    usage.append(i)
    
print(set(labels))
print(set(usage))

count0 = 0

for i, j, k in tqdm(zip(data["emotion"], data["pixels"], data["Usage"])):
    pixel = []
    pixels = j.split(' ')
    for m in pixels:
        value = float(m)
        pixel.append(value)
    pixel = np.array(pixel)
    image = pixel.reshape(48, 48)
    
    if k == "Training":
        if not os.path.exists("train"):
            os.mkdir("train")
        if i == 0:
            if not os.path.exists("train/Angry"):
                os.mkdir("train/Angry")
                path = "train/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 2:
            if not os.path.exists("train/Fear"):
                os.mkdir("train/Fear")
                path = "train/Fear/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Fear/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 3:
            if not os.path.exists("train/Happy"):
                os.mkdir("train/Happy")
                path = "train/Happy/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Happy/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 4:
            if not os.path.exists("train/Sad"):
                os.mkdir("train/Sad")
                path = "train/Sad/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Sad/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 5:
            if not os.path.exists("train/Surprise"):
                os.mkdir("train/Surprise")
                path = "train/Surprise" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Surprise/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 6:
            if not os.path.exists("train/Neutral"):
                os.mkdir("train/Neutral")
                path = "train/Neutral/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Neutral/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
    else:
        if not os.path.exists("validation"):
            os.mkdir("validation")
        if i == 0:
            if not os.path.exists("validation/Angry"):
                os.mkdir("validation/Angry")
                path = "validation/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 2:
            if not os.path.exists("validation/Fear"):
                os.mkdir("validation/Fear")
                path = "validation/Fear/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Fear/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 3:
            if not os.path.exists("validation/Happy"):
                os.mkdir("validation/Happy")
                path = "validation/Happy/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Happy/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 4:
            if not os.path.exists("validation/Sad"):
                os.mkdir("validation/Sad")
                path = "validation/Sad/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Sad/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 5:
            if not os.path.exists("validation/Surprise"):
                os.mkdir("validation/Surprise")
                path = "validation/Surprise/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Surprise/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
        if i == 6:
            if not os.path.exists("validation/Neutral"):
                os.mkdir("validation/Neutral")
                path = "validation/Neutral/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Neutral/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
               
