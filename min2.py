import numpy as np
import torch

def polynomial_function(x, params):
    """多項式関数を生成する関数"""
    y = torch.zeros_like(x)
    for i, param in enumerate(params):
        y += param * (x ** i)
    return y

def minimize_function(params, initial_value=0.0, alpha=0.0001, epochs=10000):
    """多項式関数の最小化を行う関数"""
    # パラメータをTensorに変換し、勾配計算を有効化
    params = [torch.tensor(p, requires_grad=True) for p in params]

    # 最適化ループ
    for epoch in range(epochs):
        # データの生成（例：-3から3の間のランダムな数値100個）
        rnd = np.random.uniform(-3, 3.0, 100)
        x = torch.tensor(rnd, requires_grad=True)
        
        # 関数 f(x) の計算
        y = polynomial_function(x, params)
        y_hat = polynomial_function(x, [0.5] * len(params))  # 例としての正解値
        
        # 二乗誤差の計算
        E = torch.nn.functional.mse_loss(y, y_hat, reduction='sum')
        
        # 勾配の計算
        E.backward()

        # パラメータの更新
        with torch.no_grad():
            for param in params:
                # 勾配のクリッピング
                torch.nn.utils.clip_grad_norm_(param, 1.0)
                param -= alpha * param.grad
                param.grad.zero_()
    
    # パラメータの結果を表示
    for i, param in enumerate(params):
        print(f"パラメータ p{i} = {param.data}")

    # 最小値を求めるための x の計算
    x_values = torch.linspace(-3, 3, 1000)
    y_values = polynomial_function(x_values, params)
    
    min_y, min_index = torch.min(y_values, 0)
    optimal_x = x_values[min_index]
    print(f"最小値をとる x の値: {optimal_x.data}, 最小値: {min_y.data}")

# パラメータの初期値の設定（例：2次関数 p0, p1, p2 の初期値）
initial_params = [0.1, 0.1, 0.1]
minimize_function(initial_params)

# パラメータの初期値の設定（例：3次関数 p0, p1, p2, p3 の初期値）
initial_params = [0.1, 0.1, 0.1, 0.1]
minimize_function(initial_params)

