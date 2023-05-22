import random
import numpy as np
import matplotlib.pyplot as plt

eta = 0.001

X =  [i for i in range(1,11)]
y =  [i for i in range(1,11)]
plt.figure(figsize=(10, 5))

plt.plot(X,y,'bo') 

def linearRegressionFit (X,Y,Weight,b):
    X = np.array(X)
    predicted_y= (Weight * X) + b 
    while CostFunction(Weight,predicted_y,Y) > 0.000001 :
        predicted_y= (Weight * X) + b 
        Weight = Weight - delta_weight(X,Y,predicted_y)
        b = b - delta_B(X,Y,predicted_y) 
    
    return Weight, b
    
def  linearRegressionPredict(Weight,b):
    X_Test = [i for i in range(-20,20)]
    X_Test = np.array(X_Test)
    y_Test = (Weight * X_Test) + b    
    plt.plot(X_Test,y_Test,'r-')
    plt.show()

def delta_B(X, Y, predicted_y):
    return (1 / len(X)) * sum((predicted_y - Y) )* eta

def delta_weight(X,Y,predicted_y):
    return (1 / len(X)) * np.dot ((predicted_y - Y ) , X) * eta
     
def CostFunction (Weight, predicted_y,Y):
        return (1/(2*len(X))) * sum( (predicted_y - Y )**2)

Weight = random.uniform(0,1)
b = random.uniform(0,1)
Weight,b = linearRegressionFit (X,y,Weight,b)
linearRegressionPredict(Weight,b)