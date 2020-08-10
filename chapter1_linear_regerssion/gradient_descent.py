#This is a script that hradient descent of python.
import numpy as np 
import matplotlib.pyplot as plt
#solve the minimun of y = x^2 + 2*x + 5
#draw figure of this function.
x = np.linspace(-4,2,100)
y = x**2 + 2*x + 5
plt.plot(x,y)
#init x,a,number of times
x = 3
alpha = 0.8
iteraterNum = 15

for i in range(iteraterNum):
    x = x-alpha*(2*x+2)
    plt.plot(x,x**2+2*x+5,"*")
print(x)
plt.show()