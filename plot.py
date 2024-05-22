import matplotlib.pyplot as plt
import numpy as np

content = None
with open("data_1.txt", "r") as file:
    content = file.read()

content = content.split("\n")
point_1 = [float(content[i]) for i in range(0,len(content),3)]
point_2 = [float(content[i]) for i in range(1,len(content),3)]

plt.hist(np.array(point_1))
plt.title("testing data evaluated on training data")
plt.show()

plt.hist(np.array(point_2))
plt.title("training data evaluated on training data")
plt.show()
