# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array,load_img
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import gc
import sys
import seaborn as sns

# set the matplotlib backend so figures can be saved in the background
import matplotlib

#matplotlib.use("TkAgg")
 
train_dir='D:\\keras_image_classification\\train'
test_dir='D:\\keras_image_classification\\test'
train_kill_dir='D:\\keras_image_classification\\train\\kill'
train_normal_dir='D:\\keras_image_classification\\train\\normal'
train_knocked_dir='D:\\keras_image_classification\\train\\knocked'
train_kill=[os.path.join(train_kill_dir,f) for f in os.listdir(train_kill_dir)]
train_knocked=[os.path.join(train_knocked_dir,f) for f in os.listdir(train_knocked_dir)]
train_normal=[os.path.join(train_normal_dir,f) for f in os.listdir(train_normal_dir)]
test_img=[os.path.join(test_dir,f) for f in os.listdir(test_dir)]
train_imgs=train_kill[:169]+train_knocked[:64]+train_normal[:13]
random.shuffle(train_imgs)
del train_kill
del train_knocked
del train_normal
gc.collect()
'''
测试图片能否展示
for i in train_imgs[0:3]:
    img=plt.imread(i)
    imgplot=plt.imshow(img)
    plt.show()
'''
#调整图片的大小
nrows=1920 #行数
ncolumns=1080 #列数
channels=3 #深度

def read_and_process_image(list_of_images):
    """
    返回两个数列
    第一个数列是经过调整的的图像
    第二个数列是图片的标签
    """
    X=[]
    y=[]
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))
        if 'kill' in image:
            y.append(1)
        elif 'knocked' in image:
            y.append(2)
        elif 'normal' in image:
            y.append(0)
    return X,y

X,Y=read_and_process_image(train_imgs)
#print(Y)
del train_imgs
gc.collect()
#将list转成numpy array
X=np.array(X)
Y=np.array(Y)
sns.countplot(Y)
plt.title('击杀，被击杀，普通的标签数列')
print("Shape of train images is:",X.shape)
print("Shape of labes is :",Y.shape)

#将训练数据拆分为训练集和验证集
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=2)
print("Shape of train images is:",X_train.shape)
print("Shape of validation images is:",X_val.shape)
print("Shape of lables is:",Y_train.shape)
print("Shape of labes is:",Y_val.shape)

#清理内存
del X,Y
gc.collect()

#获取训练集和验证集的长度
ntrain=len(X_train)
nval=len(X_val)

#batch_szie一般为2的倍数，不同的size效果也不一样
batch_size=8

#构建CNN神经网络模型
model=models.Sequential() #线性模型
model.add(layers.Conv2D(8,(3,4),activation='relu',input_shape=(1080,1920,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(16,(3,4),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,4),activation='relu')) #卷积层
model.add(layers.MaxPooling2D((2,2))) #最大池化层
model.add(layers.Conv2D(64,(3,4),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,4),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten()) #Flatten是一个中间层，Conv2D层提取并学习空间特征，然后在归并后将其传递给dense层，这是Flatten的工作
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#预览卷积神经网络的布局和参数大小
model.summary()

#编译模型
#binary_crossentropy二进制交叉熵损失函数，rmsprop优化器，这是超参数调优过程的一部分，metrics是测量模型性能要使用的度量标准
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

'''
ImageDataGenerator可以让我们快速设置Python生成器，自动将图像文件转换为预处理的张量，可在训练期间直接输入到模型中
他的功能有
1、将JPEG内容解码为RGB像素网格
2、将它们转换为浮点张量
3、将像素值(0~255)重新缩放到[0,1]区间
4、帮助我们轻松地增强图像

使用它时，我们创建2个生成器，一个用于训练集，一个用于验证集
'''

#创建ImageDataGenerator
train_datagen=ImageDataGenerator(
    rescale=1./255, #图片像素归一化
    
    #图像增强选项
    rotation_range=40,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

#为验证集创建一个ImageDataGenerator对象，在验证中不做数据拓展，只执行重缩放
val_datagen=ImageDataGenerator(rescale=1./255)

#传递训练集和验证集来给ImageDataGenerator对象创建Python生成器
train_generator=train_datagen.flow(X_train,Y_train,batch_size=batch_size)
val_generator=val_datagen.flow(X_val,Y_val,batch_size=batch_size)

#这一部分当数据多了之后需要修改
#训练部分
#训练16轮，给13步，这意味着我们将在整个训练集中，一次性地对模型进行总计100次梯度更新
history=model.fit_generator(
    train_generator,
    steps_per_epoch=ntrain//(batch_size*2),
    epochs=32,
    validation_data=val_generator,
    validation_steps=nval//(batch_size*2)    
    )

score=model.evaluate(X_val,Y_val)
print(score)

#保存模型
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

#绘制训练集和验证及的正确率和损失
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs = range(1,len(acc)+1)

#训练和验证的准确率
plt.plot(epochs,acc,'b',label='Training accurarcy')
plt.plot(epochs,val_acc,'r',label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()
plt.figure()

#训练和验证的损失率
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.figure()

#在测试集中的一些图像上测试模型