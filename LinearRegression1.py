import numpy as np
import matplotlib.pyplot as plt
def numerical_derivative(f,x):
    delta_x=1e-4
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        tmp_val=x[idx]
        x[idx]=float(tmp_val)+delta_x
        fx1=f(x)
        x[idx]=float(tmp_val)-delta_x
        fx2=f(x)
        grad[idx]=(fx1-fx2)/(2*delta_x)
        x[idx]=tmp_val
        it.iternext()
    return grad
def error_val(x,t):
    global W,b
    y=np.dot(x,W)+b
    return np.sum((t-y)**2)/len(x)
def predict(x):
    global W,b
    return np.dot(x,W)+b
loaded_data=np.loadtxt("./data-01-test-score.csv",delimiter=',',dtype=np.float32)
x_data=loaded_data[:,0:-1]
t_data=loaded_data[:,[-1]]
W=np.random.rand(x_data.shape[1],t_data.shape[1])
b=np.random.rand(1)
f=lambda x:error_val(x_data,t_data)
learning_late=1e-5
for step in range(8001):
    W=W-learning_late*numerical_derivative(f,W)
    b=b-learning_late*numerical_derivative(f,b)
    if (step % 400 == 0):
        print("step=", step, "error value=", error_val(x_data, t_data), "W=", W, "b=", b, )
