import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle
import pandas



def Remove_outliers(data, threshold=3):
  mean = numpy.mean(data) # 求期望
  std = numpy.std(data)  # 求方差
  # 以下两行数据清洗
  score = [(x - mean) / std for x in data]  
  # 呃简单理解就是把杂乱分布的数据x变为标准正态分布中的点score
  filtered_data = [x if abs(z) < threshold else 0 for x,z in zip(data, score)]
  # 简单理解就是只取正态分布中间某个范围内的点，abs指绝对值，
  # score绝对值小于3则保留，大于3则用0替代x
  
  """
  # 上两行数据清洗也可以用以下方式实现：
  filtered_data = []
  for x in data:
    score = (x - mean) / std
    if abs(score) < threshold:
      filtered_data.append(x)
    else:
      filtered_data.append(0)  
  """
  return numpy.interp(numpy.arange(len(filtered_data)), numpy.flatnonzero(filtered_data), numpy.array(filtered_data)[numpy.flatnonzero(filtered_data)])
# interp(x,xp,fp): 线性插值;x：待插入数据的横坐标，xp：一维浮点数序列，fp：原始数据点的纵坐标
# arange:返回一个有终点和起点的固定步长的排列
# flatnonzero:输入一个矩阵，返回了其中非零元素的位置.
# 大概意思就是把填0的地方线性插值一下

# Read Excel Files
# 调用python的pandas代码库来读取磁场数据，但matlab不一定是同样的格式
data = pandas.read_excel(r"E:\Users\Admin\Desktop\日程表\202306\空间天气预报作业\磁场数据7天.xls")
train = data["Hn"] # 调取Hn数据
train = numpy.array(train) # 转成数组

# Remove Outliers
train = Remove_outliers(train)  # 调用上面定义的函数来清除异常点
train_print = train[0:5] # 尝试截取前五行打印
print(train_print)

train_len = len(train)  # 计算数列的长度，作为绘图的横坐标
print(train_len)  # out: 10079
print(max(train),min(train))
# plt.plot(numpy.arange(len(train)),train)
# plt.show()


# 把train变为二维数组
# 因为后续的神经网络输入需要三维
train = train.reshape(-1, 1)
max_train = np.max(train, axis=0)
min_train = np.min(train, axis=0)
print(max_train, min_train)
# out: [19.04428864] [-23.61950874]

# nomalization
# 归一化，把train里所有数据变成(0,1)区间
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
train = min_max_scaler.fit_transform(train)
with open("min_max_scaler.pkl","wb") as f:
  pickle.dump(min_max_scaler, f)

print(train[:,0])
# out: [5.38424063 5.40454245 5.59618568 ... 1.64781845 1.62875116 1.61578417]
# plt.plot(numpy.arange(train_len),train[:,0])
# plt.show()

# Create GRU Network
model = Sequential()
model.add(GRU(units=100, activation='relu', input_shape=(100,1)))
model.add(Dense(units=10))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

ccc = 10079-100-10+1
new_X_train = numpy.zeros((ccc, 100, 1))
new_Y_train = numpy.zeros((ccc, 10))

# 每隔100个元素，切片做成一个数组，将数组赋值new_X_train的一个元素里
# 从100+i 开始，每隔10个元素，将其切片做成数组（二维，后面赋值0），赋值给new_Y_train的一个元素上
for i in range(ccc):
  new_X_train[i] = train[i:i+100]

for i in range(ccc):
  new_Y_train[i] = train[100+i:100+i+10, 0 ]
# 大概意思就是用前一百个数据预测后十个数据
  
model.fit(new_X_train[0:9000], new_Y_train[0:9000], epochs=100, batch_size=128)

plt.plot(model.history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale("log")
plt.show()

# 显示训练组预测值和准确值之间的误差
outputs = model(new_X_train[0:9000])
yy=new_Y_train[0:9000]
outputs=np.array(outputs)
print(outputs,yy)

plt.plot(np.arange(90000),outputs.reshape(-1),label='预测值')
plt.plot(np.arange(90000),yy.reshape(-1),label='准确值')
 
plt.legend()

 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 显示预测组 预测值与准确值之间的误差
outputs = model(new_X_train[9000:9970])
yy=new_Y_train[9000:9970]
outputs=np.array(outputs)
print(outputs,yy)

plt.plot(np.arange(9700),outputs.reshape(-1),label='预测值')
plt.plot(np.arange(9700),yy.reshape(-1),label='准确值')
 
plt.legend()

 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 输出预测值
Y_=[]
train2=train.reshape(-1)
for i in np.arange(144):
    X_=train2[-100:].copy()
    X_=X_.reshape(1,-1,1)
    outputs = model(X_)
    outputs=np.array(outputs).reshape(-1)
    train2=np.concatenate((train2,outputs),axis=0)
    print(train2.shape)
    Y_=np.concatenate((Y_,outputs),axis=0)

# 从归一化复原
Y_2=Y_*(max_values - min_values) + min_values

# 画预测出的磁场一天图
plt.plot(np.arange(1440),Y_2,label='预测值')

# 添加横纵标题
plt.title('磁场数据1天预测')
plt.xlabel('时间/t')
plt.ylabel('磁场强度/Hn')
 
plt.legend()

 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
