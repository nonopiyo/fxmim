import numpy as np
import torch

alpha = 0.001

unit = torch.nn.linear(1,1,bias = True)

for epoch in range (1000):
    rnd = np.random.uniform(-3,3.0,100) #low,high,何個
    x = torch.tensor(rnd, requires_grad=True).reshape(-1,1)
    y = a * x + b
    y_hat = 0.5 * x + 0.3
    E = 0.5 * (y - y_hat) ** 2  #MSE
    E = torch.nn.functional.mse_loss(y,y_hat,reduction="sum")

    a.retain_grad()
    b.retain_grad()
    E.backward()
    unit.update_parameters(alpha)

print(f"a = {unit.weight.data}, b = {unit.bias.data}")
