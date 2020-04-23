import numpy as np
from sklearn.datasets import load_boston

# 数据加载
data = load_boston()
x_ = data['data']
y = data['target']

# 将y转化为矩阵的形式
y = y.reshape(y.shape[0],1)   # 数据升维
# print(x.shape,y.shape) # (506, 13) (506, 1)


# 数据规范化 (正态分布)
x_ = (x_ - np.mean(x_, axis=0))/np.std(x_, axis=0)


n_features = x_.shape[1]
n_hidden = 10

# 初始化网络参数,定义隐藏层维度，w1,b1,w2,b2
w1 = np.random.randn(n_features,n_hidden)
b1 = np.zeros(n_hidden)
w2 = np.random.rand(n_hidden,1)
b2 = np.zeros(1)

# 激活函数Relu
def Relu(x):
    result = np.where(x < 0, 0, x)
    return result
# 损失函数MSE
def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))
# 设置学习率
learning_rate = 1e-6

# 线性回归函数Linear
def Linear(x, w1, b1):
    y = x.dot(w1) + b1
    return y

# 设置迭代次数
for t in range(5000):
    # 前向传播
    l1 = Linear(x_, w1, b1)
    s1 = Relu(l1)
    y_pred = Linear(s1, w2, b2)

    # 计算损失函数
    loss = MSE_loss(y,y_pred)
    # print(t, loss)

    # 反向传播
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = s1.T.dot(grad_y_pred)
    grad_temp_relu = grad_y_pred.dot(w2.T)
    grad_temp_relu[l1<0] = 0
    grad_w1 = x_.T.dot(grad_temp_relu)

    # 权重更新
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print(f'w1={w1}\n w2={w2}')




