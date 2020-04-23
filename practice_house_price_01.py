import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([1,3,2,3,5])
plt.scatter(x,y)
plt.grid()
# plt.show()

#定义损失函数
def MAE_loss(y, y_hat):
    return np.mean(np.abs(y_hat - y))

def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))
y_hat = np.array([-2,-1,-1,3,5])
print(MAE_loss(y,y_hat))

#定义线性回归函数
def linear(x,k,b):
    y = k*x+b
    return y
min_loss = float('inf')
for k in np.arange(-2,2,0.1):
    for b in np.arange(-10,10,0.1):
        y_hat = [linear(xi,k,b) for xi in list(x)]
        # current_loss = MAE_loss(y,y_hat)
        current_loss = MSE_loss(y, y_hat)
        if current_loss < min_loss:
            min_loss = current_loss
            best_k, best_b = k, b
            print(f'best_k is {best_k},best_b is {best_b}')

# best_k = 0.8
# best_b = 0.4
y_hat = best_k * x + best_b
plt.plot(x,y_hat,color='red')
plt.grid()
plt.show()


