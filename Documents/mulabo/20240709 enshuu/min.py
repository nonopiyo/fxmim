import numpy as np
import torch

# 学習率とエポック数の設定
alpha = 0.001
epochs = 1000

# パラメータの初期化
a = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

# 勾配降下法のループ
for epoch in range(epochs):
    # データの生成（例：-3から3の間のランダムな数値100個）
    rnd = np.random.uniform(-3, 3.0, 100)
    x = torch.tensor(rnd, requires_grad=True)
    
    # 関数 f(x) = a * x + b の計算
    y = a * x + b
    y_hat = 0.5 * x + 0.3  # 例としての正解値（ここでは y_hat を変更しない）
    
    # 二乗誤差の計算
    E = torch.nn.functional.mse_loss(y, y_hat, reduction='sum')
    
    # 勾配の計算
    E.backward()

    # パラメータの更新
    with torch.no_grad():
        a -= alpha * a.grad
        b -= alpha * b.grad
    
    # 勾配の初期化
    a.grad.zero_()
    b.grad.zero_()

print(f"a = {a.data}, b = {b.data}")

# 最小値を求めるための x の計算
# y = 0 のときの x を求める（a * x + b = 0 を解く）
optimal_x = -b / a
print(f"最小値をとる x の値: {optimal_x.data}")

