#-*- coding: UTF-8 -*-
import os,sys
import tensorflow as tf
import numpy as np


output_dim=260
input_dim=39
learning_rate=0.001

def REdistinct(files):
    distinct=[]
    for i in files:
        if i.split('.')[0] not in distinct:
            distinct.append(i.split('.')[0])
    return distinct
def RENumpyData(distinct,dir="./train"):
    feature=[]
    target=[]
    for i in distinct:
        featureData=np.loadtxt(dir+'/'+i+".mfcc", dtype=float).astype("float")
        featureDataFrame,featureDataDim=featureData.shape
        targetData=np.loadtxt(dir+'/'+i+".phone", dtype=float).astype("float")
        targetDataFrame,=targetData.shape
        assert featureDataDim==input_dim
        assert featureDataFrame==targetDataFrame
        featureData=np.reshape(featureData,(1,featureDataFrame,featureDataDim))
        feature.append(featureData)
        targetData=tf.one_hot(targetData,output_dim,1,0)
        target.append(targetData)
    return feature,target






train_path="./train"
trainfile=os.listdir(train_path)
trainfile.sort()
distinct=REdistinct(trainfile)

TrainFeature,TrainTarget=RENumpyData(distinct)


input=tf.placeholder(tf.float32,[1,None,input_dim])
y=tf.placeholder(tf.float32,[None,output_dim])
#这样做是因为在conv1d中，stride是一维的，因此只能将行数变为1，在列数上stride，同时，深度为特征的维数
#frame:11,feature:8,depth:1
h1_kernel_size = 1
h1_filters = 500
#h1_filters:output_dim
h1 = tf.layers.conv1d(input, h1_filters, h1_kernel_size,use_bias=True)
h1_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h1))


#h1_kernel_size:一次性拿多少时间点
h2_kernel_size = 1
h2_filters=500
h2=tf.layers.conv1d(h1_relu_renorm,h2_filters,h2_kernel_size,use_bias=True)
h2_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h2))

h3_kernel_size = 1
h3_filters=500
h3=tf.layers.conv1d(h2_relu_renorm,h3_filters,h3_kernel_size,use_bias=True)
h3_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h3))

h4_kernel_size = 1
h4_filters=500
h4=tf.layers.conv1d(h3_relu_renorm,h4_filters,h4_kernel_size,use_bias=True)
h4_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h4))

h5_kernel_size = 1
h5_filters=500
h5=tf.layers.conv1d(h4_relu_renorm,h5_filters,h5_kernel_size,use_bias=True)
h5_relu_renorm=tf.layers.batch_normalization(tf.nn.relu(h5))

output_kernel_size=1
output_filter=output_dim
output=tf.layers.conv1d(h5_relu_renorm,output_filter,output_kernel_size,use_bias=True)
output_softmax=tf.nn.softmax(output)

loss= tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.squeeze(output_softmax)), reduction_indices=[1]))
#loss=tf.reduce_mean(tf.square(tf.squeeze(output_softmax)-y))
#loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(output_softmax)))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)



sess=tf.Session()
sess.run(tf.global_variables_initializer())


for j in range(1,31):
    pre=0
    for i in range(0,len(distinct)):
        _,a=sess.run([train_op,loss],feed_dict={input:TrainFeature[i],y:TrainTarget[i].eval(session=sess)})
        pre=a+pre
    print pre/len(distinct)
    if j % 5 ==0:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.save(sess,'./model'+str(j))





