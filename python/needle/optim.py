"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data + w.data * self.weight_decay
            if self.u.get(w) is None:
                self.u[w] = (1 - self.momentum) * grad
            else:
                self.u[w] = (self.momentum * self.u[w] + (1 - self.momentum) * grad).detach()

            w.data = w.data - (self.lr * ndl.Tensor(self.u[w].numpy().astype(np.float32))).data   

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for w in self.params:
            if w.grad is None:
                continue
            grad = w.grad.data + w.data * self.weight_decay
            if self.m.get(w) is None:
                self.m[w] = ((1 - self.beta1) * grad).detach()
                self.v[w] = ((1 - self.beta2) * grad * grad).detach()
            else:
                self.m[w] = (self.beta1 * self.m[w] + (1 - self.beta1) * grad).detach()
                self.v[w] = (self.beta2 * self.v[w] + (1 - self.beta2) * grad * grad).detach()

            m_with_hat = self.m[w].data / (1 - self.beta1 ** self.t)
            v_with_hat = self.v[w].data / (1 - self.beta2 ** self.t)
            
            # print(np.linalg.norm(v_with_hat.numpy()), np.linalg.norm((v_with_hat ** (0.5)).numpy()))
            # print(self.beta1, self.beta2, self.t, np.linalg.norm(w.grad.numpy()), np.linalg.norm(w.numpy()))
            
            w.data = w.data - (self.lr * ndl.Tensor((m_with_hat.numpy() / (v_with_hat ** (0.5) + self.eps).numpy()).astype(np.float32))).data
