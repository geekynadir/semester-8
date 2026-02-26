import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
ch=int(input("Enter the choice between [1 to 5]"))
if (ch==1):
    print("Addition:", a + b)
elif (ch==2):
    print("Subtraction:", a - b)
elif (ch==3):
    print("Multiplication:", a * b)
elif (ch==4):
       print("Division:", a / b)
else:
    print("Power:", a ** 2)
    





