"""The module.
"""
from doctest import OutputChecker
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as array_api

from python import needle
from python.needle.ops import EWiseAdd


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        self.y = x
        return x
    def backward(self):
        self.y.backward()
        return self.y.grad


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose()) if bias else None
          
    def forward(self, X: Tensor) -> Tensor:
        self.X = X

        if self.bias:
            if len(self.weight.shape) == 1:
                return X @ self.weight + self.bias.broadcast_to((X.shape[0], 1))
            else:
                return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.weight.shape[-1]))
        return X @ self.weight

# class Linear(Module):
#     def __init__(self, in_features, out_features, bias=True, device=None, dtype="float64"):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
#         self.has_bias = bias

#         if bias:
#             self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose())

#     def forward(self, X: Tensor) -> Tensor:
#         batch_size = X.shape[0]
#         logits = X @ self.weight

#         if self.has_bias:
#             reshaped_bias = ops.broadcast_to(self.bias, 
#                                             (batch_size, self.out_features))
#             logits += reshaped_bias

#         return logits

#     def backward(self):
#         if self.X.grad is not None:
#             return self.forward(self.x).backward(self.X.grad)

#         return self.forward(self.x).backward()

class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], -1))

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for module in self.modules:
            y = module(y)
        return y

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        Z = needle.logsumexp(logits, 1)

        y_one_hot = init.one_hot(logits.shape[1], y)

        Z_y = logits * y_one_hot
        return (EWiseAdd()(Z, -Z_y.sum(1))).sum() / Z.shape[0]

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.bias = Parameter(init.zeros(*(1, dim)))
        self.weight = Parameter(init.ones(*(1, dim)))

        self.running_mean = init.zeros(dim, device=device, dtype=dtype).data
        self.running_var = init.ones(dim, device=device, dtype=dtype).data

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            X_mean = x.mean(axes=(0, ), keepdims=True).broadcast_to(x.shape)
            X_var = ((x - X_mean)**2).mean(axes=(0, ), keepdims=True).broadcast_to(x.shape)
            # X_var = x.var(axes=(0, ), keepdims=True).broadcast_to(x.shape)

            bnorm = (x - X_mean) / (X_var + self.eps)**(1/2)
            y = self.weight.broadcast_to(bnorm.shape) * bnorm + self.bias.broadcast_to(bnorm.shape)
            
            run_x_mean = x.mean(axes=(0, )).data
            run_x_var = ((x - run_x_mean)**2).mean(axes=(0, )).data

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * run_x_mean)
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * run_x_var)

            return y
        else:
            bnorm = (x - self.running_mean) / (self.running_var + self.eps)**(1/2)
            y = self.weight.data.broadcast_to(bnorm.shape) * bnorm + self.bias.data.broadcast_to(bnorm.shape)
            return y


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.bias = Parameter(init.zeros(*(1, dim)))
        self.weight = Parameter(init.ones(*(1, dim)))

    def forward(self, x: Tensor) -> Tensor:
        X_mean = x.mean(axes=(1, ), keepdims=True).broadcast_to(x.shape)
        X_var = ((x - X_mean)**2).mean(axes=(1, ), keepdims=True).broadcast_to(x.shape)

        _ = x.mean(axes=(1, ), keepdims=True)
        lnorm = (x - X_mean) / (X_var + self.eps)**(1/2)
        y = self.weight.broadcast_to(lnorm.shape) * lnorm + self.bias.broadcast_to(lnorm.shape)
        return y

class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x * init.randb(*x.shape, p=1 - self.p) / (1-self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x