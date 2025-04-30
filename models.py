from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        # 网络结构
        self.conv1 = conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()

        self.conv2 = conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()

        # 输入图像为 28x28，经过两层 padding=1 的卷积后大小不变，最后展开为 16*28*28
        self.fc = Linear(in_dim=16 * 28 * 28, out_dim=10)

        # 可迭代调用的层列表（方便 forward/backward 循环）
        self.layers = [self.conv1, self.relu1, self.conv2, self.relu2, self.fc]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers[:-1]:  # 最后一层 Linear 之前
            out = layer(out)

        # 手动 flatten
        B = out.shape[0]
        out = out.reshape(B, -1)

        out = self.fc(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        # 因为 Linear 层前进行了 flatten，要先 reshape 再传回去
        grad = self.fc.backward(grad)

        # reshape 为 conv 输出形状
        grad = grad.reshape(-1, 16, 28, 28)

        for layer in reversed(self.layers[:-1]):  # Linear 已处理
            grad = layer.backward(grad)
    
    def load_model(self, param_list):
        for layer, param in zip(self.layers, param_list):
            if hasattr(layer, 'params') and param is not None:
                for key in layer.params:
                    layer.params[key][...] = param[key]
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                param_list.append({k: v.copy() for k, v in layer.params.items()})
            else:
                param_list.append(None)
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)