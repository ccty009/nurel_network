from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                # 直接修改权重和偏置
                if 'W' in layer.params:
                    layer.W -= self.init_lr * layer.grads['W']
                if 'b' in layer.params:
                    layer.b -= self.init_lr * layer.grads['b']


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu  # 动量因子
        self.velocity = {}  # 存储动量

        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[layer] = {}
                for key in layer.params:
                    self.velocity[layer][key] = np.zeros_like(layer.params[key])

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    grad = layer.grads.get(key)

                    # 跳过 None 梯度
                    if grad is None:
                        continue

                    # 权重衰减作为梯度项添加
                    if layer.weight_decay:
                        grad = grad + layer.weight_decay_lambda * layer.params[key]

                    # 确保 param 是 float 类型
                    if not np.issubdtype(layer.params[key].dtype, np.floating):
                        layer.params[key] = layer.params[key].astype(np.float32)

                    # 获取旧动量并更新
                    v_old = self.velocity[layer][key]
                    v_new = self.mu * v_old - self.init_lr * grad

                    # 更新参数
                    layer.params[key] += v_new

                    # 保存新的动量
                    self.velocity[layer][key] = v_new

                    # 清零梯度
                    layer.grads[key] = np.zeros_like(layer.grads[key])