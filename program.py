import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# creating net
net = nn.Sequential(nn.Linear(15, 32, bias=True), nn.ReLU(),
                    nn.Linear(32, 64, bias=True), nn.ReLU(),
                    nn.Linear(64, 15, bias=True))

for prog in net:
    if isinstance(prog, torch.nn.Linear):
        nn.init.normal_(prog.weight, mean=1.0, std=2.0)

# generate training set
# generating noise
std, mean = 1.0, 3.0
noise_mat = torch.normal(mean, std, (1, 15))
print(noise_mat)

# generating data couple
X_no_noise = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
X_no_noise.resize(1, 15)
X = X_no_noise + noise_mat
X.reshape(1, 15)
print(X)

# ideal output y
# we suppose the equation to be y = 2.1 + 3.2 * x1 + 1.6 * x1 ** 2 + 0.9 * x1 ** 3
def f(x1):
    return 2.1 + 3.2 * x1 + 1.6 * x1 ** 2 + 0.9 * x1 ** 3
Y = f(X_no_noise)
Y.reshape(1, 15)

# after learning that the noise is normal distribution-ish, we substract the noise from the input.
# Since such learning is not completely accurate, we suppose that the normal distribution to be mean = 2.8, std = 0.9;
mean_learned = 2.8
std_learned = 0.9
anti_noise_mat = torch.normal(mean_learned, std_learned, (1, 15))
X_optimized = X - anti_noise_mat


# training net with X
epochs, lr = 150, 0.1

loss = nn.MSELoss()
opt = torch.optim.SGD(net.parameters(), lr=lr)

graphx_1 = []
graphy_1 = []
graphx_2 = []
graphy_2 = []

# ordinary
for epoch in range(epochs):
    opt.zero_grad()
    Y_prime = net(X)
    l = loss(Y, Y_prime)
    l.backward()
    opt.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {l.item():.4f}')
        graphx_1.append(epoch+1)
        graphy_1.append(l.item())
        print (Y - net(X))

# optimized

for epoch in range(epochs):
    opt.zero_grad()
    Y_prime = net(X_optimized)
    l = loss(Y, Y_prime)
    l.backward()
    opt.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {l.item():.4f}')
        graphx_2.append(epoch+1)
        graphy_2.append(l.item())
        print(Y - net(X_optimized))

# visualize the training process
plt.plot(graphx_1, graphy_1, label='Loss before', color='black', linestyle='--')
plt.plot(graphx_2, graphy_2, label='Loss after', color='red')

plt.title("loss graph")
plt.xlabel("epoch number")
plt.ylabel("loss")
plt.show()






