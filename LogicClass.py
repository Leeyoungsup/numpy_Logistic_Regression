import numpy as np
import matplotlib.pyplot as plt
class LogicGate:
    def __init__(self,gate_name,xdata,tdata):
        self.name=gate_name
        self.__xdata=xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4, 1)
        self.__W=np.random.rand(self.__xdata.shape[1],self.__tdata.shape[1])
        self.__b = np.random.rand(1)
        self.__learning_rate=1e-2

    def __loss_func(self):
        delta = 1e-7
        z = np.dot(self.__xdata,self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))
    def train(self):
        f = lambda x: self.__loss_func()
        for step in range(8001):
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if (step % 400 == 0):
                print("step=", step, "error value=", self.error_val(), "W=", self.__W, "b=", self.__b, )
    def error_val(self):
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum(self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta))

    def predict(self,x):
        z = np.dot(x, self.__W) + self.__b
        y = sigmoid(z)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y,result

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
def sigmoid(x):
    return 1/(1+np.exp(-x))
x_data=np.array([[0,0],[1,0],[0,1],[1,1]])
t_data=np.array([1,1,1,0])
y_data=np.zeros_like(x_data)
And_obj=LogicGate("Nand_GATE",x_data,t_data)
And_obj.train()
test_data=np.array([[0,0],[0,1],[1,0],[1,1]])
count=0
for input_data in test_data:
    (sigmoid_val,logical_val)=And_obj.predict(input_data)
    y_data[count,0]=logical_val
    print(sigmoid_val,logical_val)
    count+=1
 
x_data=np.array([[0,0],[1,0],[0,1],[1,1]])
t_data=np.array([0,1,1,1])
And_obj=LogicGate("Or_GATE",x_data,t_data)
And_obj.train()
test_data=np.array([[0,0],[0,1],[1,0],[1,1]])
count=0
for input_data in test_data:
    (sigmoid_val,logical_val)=And_obj.predict(input_data)
    y_data[count,1]=logical_val
    print(sigmoid_val,logical_val)
    count+=1
x_data=np.array([[0,0],[1,0],[0,1],[1,1]])
t_data=np.array([0,0,0,1])
And_obj=LogicGate("And_GATE",x_data,t_data)
And_obj.train()
test_data=np.copy(y_data)
for input_data in test_data:
    (sigmoid_val,logical_val)=And_obj.predict(input_data)
    print(sigmoid_val,logical_val)
