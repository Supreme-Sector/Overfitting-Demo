import linear_network, relu_network, softplus_network
import data_retriever
import matplotlib.pyplot as plt
import numpy as np

training_data = data_retriever.get_data()

data_size = len(training_data)

net1 = linear_network.LinearNetwork([1, 1])
# net2 = relu_network.ReluNetwork([1, 5, 7, 7, 7, 1]) # ReLU network was not used
net2 = softplus_network.SoftplusNetwork([1, 5, 7, 7, 7, 1])


# For training, I increased the number of epochs from 800 to 1500
# to give the network more time to fit the data
net1.SGD(training_data, 1500, data_size, 0.007)
net2.SGD(training_data, 1500, data_size, 0.007)

x = [0.1*i for i in range(-40, 110)]
y_linear = [net1.feedforward(np.array([[i]]))[0][0] for i in x]
y_softplus = [net2.feedforward(np.array([[i]]))[0][0] for i in x]

x_data = [x[0][0] for (x, y) in training_data]
y_data = [y[0][0] for (x, y) in training_data]

plt.scatter(x_data, y_data, color="black")
plt.plot(x, y_linear, color="blue")
plt.plot(x, y_softplus, color="orange")
plt.show()
