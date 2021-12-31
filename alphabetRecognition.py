import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if(not os.environ.get("PYTHONHTTPSVERIFY","")and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context=ssl._create_unverified_context
X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
classes=["A","B","C","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
xTrain,xTest,yTrain,yTest=train_test_split(X,y,train_size=7500,test_size=2500,random_state=9)
xTrainScaled=xTrain/255.0
xTestScaled=xTest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)
yPred=clf.predict(xTestScaled)
accuracy=accuracy_score(yTest,yPred)
print(accuracy)
cap=cv2.VideoCapture(0)
while True:
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert("L")
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixelFilter = 20
        minPixel=np.percentile(image_bw_resized_inverted,pixelFilter)
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-minPixel,0,255)
        maxPixel=np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/maxPixel
        testSample=np.array(image_bw_resized_inverted_scaled).ratio.reshape(1,784)
        testPred=clf.predict(testSample)
        print("predicted class is ",testPred)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1):
            break
    except:
        pass