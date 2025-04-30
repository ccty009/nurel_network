from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """

    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # Save input for backward
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # Gradients of loss w.r.t. weights and biases
        dW = self.input.T @ grad  # shape: [in_dim, out_dim]
        db = np.sum(grad, axis=0, keepdims=True)  # shape: [1, out_dim]

        # Apply weight decay if enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        # Store gradients
        self.grads['W'] = dW
        self.grads['b'] = db

        # Return gradient for previous layer
        return grad @ self.W.T  # shape: [batch_size, in_dim]
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # 卷积核 W: [out_channels, in_channels, k, k]
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.b = np.zeros((out_channels, 1))

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X  # [B, C_in, H, W]
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # padding
        if p > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            X_padded = X

        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        out = np.zeros((B, self.out_channels, H_out, W_out))

        # 卷积操作
        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = X_padded[b, :, h_start:h_start+k, w_start:w_start+k]
                        out[b, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]

        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding
        H_out, W_out = grads.shape[2:]

        # 初始化梯度
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)

        # padding
        if p > 0:
            X_padded = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
            dX_padded = np.pad(dX, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            X_padded = X
            dX_padded = dX

        for b in range(B):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        region = X_padded[b, :, h_start:h_start+k, w_start:w_start+k]

                        dW[oc] += grads[b, oc, i, j] * region
                        db[oc] += grads[b, oc, i, j]
                        dX_padded[b, :, h_start:h_start+k, w_start:w_start+k] += grads[b, oc, i, j] * self.W[oc]

        # 处理权重衰减
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        # 去掉 padding
        if p > 0:
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.preds = None  
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes_all_try here ----/
        self.labels = labels
        if self.has_softmax:
            # 防止数值爆炸：减去最大值再 softmax
            exps = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
            self.preds = exps / np.sum(exps, axis=1, keepdims=True)
        else:
            self.preds = predicts  

        batch_size = labels.shape[0]
        # 取出对应的概率值（即正确类别的概率）
        correct_probs = self.preds[np.arange(batch_size), labels]
        # 交叉熵损失
        loss = -np.mean(np.log(correct_probs + 1e-12))  # 加 epsilon 避免 log(0)

        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes_all_try here ----/
        batch_size = self.labels.shape[0]
        grad = self.preds.copy()
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size  # 平均损失的梯度

        self.grads = grad
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_=1e-4):
        super().__init__()
        self.model = model      # 需要关联的模型
        self.lambda_ = lambda_  # 正则化系数
        self.optimizable = False
        self.reg_loss = 0.0     # 记录当前的正则化损失值

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 计算所有Linear层的L2正则化损失（不影响数据流）
        self.reg_loss = 0.0
        for layer in self.model.layers:
            if isinstance(layer, Linear) and layer.optimizable:
                self.reg_loss += 0.5 * self.lambda_ * np.sum(layer.W ** 2)
        return X  # 原样传递输入数据

    def backward(self, grad):
        # 为所有Linear层的权重添加L2梯度
        for layer in self.model.layers:
            if isinstance(layer, Linear) and layer.optimizable:
                if 'W' in layer.grads:
                    layer.grads['W'] += self.lambda_ * layer.W
        return grad  # 原样传递梯度


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p          # 丢弃概率
        self.mask = None    # 二值掩码
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:  # 仅在训练时应用Dropout
            self.mask = (np.random.rand(*X.shape) > self.p)
            return X * self.mask / (1 - self.p)  # 缩放激活值以保持期望不变
        return X

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1 - self.p)
        return grad


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p          # 丢弃概率
        self.mask = None    # 二值掩码
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training:  # 仅在训练时应用Dropout
            self.mask = (np.random.rand(*X.shape) > self.p)
            return X * self.mask / (1 - self.p)  # 缩放激活值以保持期望不变
        return X

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1 - self.p)
        return grad
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition