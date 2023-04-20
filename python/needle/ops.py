"""Operator table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy
from .init import one_hot
# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        X = node.inputs[0]
        return self.scalar * out_grad * (X**(self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        X, Y = node.inputs

        return (out_grad / Y, -X * out_grad / Y ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (-2, -1) if not axes else axes

    def compute(self, a):
        return array_api.swapaxes(a, *self.axes)
        
    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        shape = node.inputs[0].shape
        return out_grad.reshape(shape)


def reshape(a, shape):
    return Reshape(shape)(a)

def get_output_shape(arr: tuple, subarr: tuple) -> List[bool]:
        if len(subarr) == 0:
            return tuple([0]*len(arr))
        main_str = ' '.join([str(i) for i in arr])
        sub_str = ' '.join([str(i) for i in subarr])

        pos = main_str.find(sub_str)

        first_iter = '0 ' * main_str[:pos+1].count(" ") + sub_str
        final_iter = first_iter + ' 0'*(main_str.count(" ") - first_iter.count(" "))

        return tuple(map(int, final_iter.split(" ")))

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        X = node.inputs[0]
        # print(X.shape, self.shape)
        
        after_shape = array_api.asarray(self.shape, dtype=int).reshape(1, -1)

        bef_tuple = get_output_shape(self.shape, X.shape)
        
        bef_shape = array_api.asarray(bef_tuple, dtype=int).reshape(1, -1)

        # print(bef_shape, after_shape)
        mask = (bef_shape != after_shape)
        # print(bef_shape, after_shape)
        # print(mask, "<-MASK")
        mask = mask[0] 
        
        delta = tuple(array_api.arange(len(bef_tuple))[mask])               
        
        if len(X.shape) == 1:
            return out_grad.sum(axes=delta).reshape((X.shape[0], -1))
        return out_grad.sum(axes=delta).reshape(X.shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Variation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        self.axes = axes
        self.keepdims = keepdims
 
    def compute(self, a):
        result = array_api.var(a, self.axes)
 
        if self.keepdims:
            outp_shape = get_unsq_outp_shape(list(a.shape), self.axes)
            return result.reshape(outp_shape)
        else:
            return result
 
    def gradient(self, out_grad, node):
        # print("Variation backward:", out_grad, node)
        # print("Shapes:", out_grad.shape, node.inputs[0].shape)
        inp = node.inputs[0]
        inp_shape = list(inp.shape)
 
        inp_mean = inp.mean(self.axes, keepdims=True)
        inp_mean = inp_mean.broadcast_to(inp_shape)
 
        reduced_size = inp.size if self.axes is None \
                        else numpy.product([inp.shape[ax] for ax in self.axes])
        grad_coeff = 2. / reduced_size
 
        broadcasted_grad = Summation(self.axes, self.keepdims).gradient(out_grad, node)
 
        return grad_coeff * broadcasted_grad * (inp - inp_mean)
 
 
def variation(a, axes=None, keepdims=False):
    return Variation(axes, keepdims)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        if axes is not None:
            self.axes = axes if type(axes) == tuple else (axes, )
        else:
            self.axes = None
        self.keepdims = keepdims

    @staticmethod
    def get_output_shape(inp_shape: List[int], axes: Optional[tuple] = None):
        outp_shape = inp_shape.copy()

        if axes is None:
            outp_shape[:] = [1] * len(inp_shape)
        else:
            for ax in axes:
                outp_shape[ax] = 1

        return outp_shape

    def compute(self, a):
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        inp_shape = list(node.inputs[0].shape)
        out_shape = Summation.get_output_shape(inp_shape, self.axes)
        
        if self.keepdims:
            inp_axis = node.inputs[0].shape
            return broadcast_to(out_grad, inp_axis)
        return broadcast_to(reshape(out_grad, out_shape), inp_shape)

        
    
def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)

def sqrt(a):
    return PowerScalar(0.5)(a)

# class Summation(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         if type(axes) == int:
#             self.axes = (axes, )
#         else:
#             self.axes = axes

#     @staticmethod
#     def get_output_shape(inp_shape: List[int], axes: Optional[tuple] = None):
#         outp_shape = inp_shape.copy()

#         if axes is None:
#             outp_shape[:] = [1] * len(inp_shape)
#         else:
#             for ax in axes:
#                 outp_shape[ax] = 1

#         return outp_shape

#     def compute(self, a):
#         if self.axes is not None:
#             return array_api.sum(a, self.axes)
#         else:
#             return a

#     def gradient(self, out_grad, node):
#         if self.axes is not None:
#             inp_axis = node.inputs[0].shape
#             return (broadcast_to(Tensor(array_api.expand_dims(out_grad.numpy(), self.axes)), inp_axis), )

#         return out_grad


# def summation(a, axes=None):
#     return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        l_shape_in = lhs.realize_cached_data().shape
        r_shape_in = rhs.realize_cached_data().shape

        out1, out2 = out_grad @ rhs.transpose(),  lhs.transpose() @ out_grad

        l_shape_out = out1.realize_cached_data().shape
        r_shape_out = out2.realize_cached_data().shape

        if len(l_shape_in) != len(l_shape_out):
            out1 = Tensor(
                out1.realize_cached_data().sum(
                    tuple(
                        range(len(l_shape_out) - len(l_shape_in))
                    )
                )
            )

        if len(r_shape_in) != len(r_shape_out):
            out2 = Tensor(
                out2.realize_cached_data().sum(
                    tuple(
                        range(len(r_shape_out) - len(r_shape_in))
                    )
                )
            )
        return out1, out2


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)

class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return Tensor((node.inputs[0].numpy() > 0) * out_grad.numpy(), device=out_grad.device)

def relu(a):
    return ReLU()(a)

class Mean(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        self.axes = axes if isinstance(axes, tuple) else (axes, )
        self.keepdims = keepdims

    def compute(self, a):
        if self.axes is not None:
            return array_api.mean(a, self.axes, keepdims=self.keepdims)
        else:
            return array_api.mean(a, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        inp_size = numpy.product(node.inputs[0].shape)
        grad_size = 1 if out_grad.shape == tuple() \
                    else numpy.product(out_grad.shape)
        size_diff = inp_size / grad_size

        sum_grad = Summation(self.axes, self.keepdims).gradient(out_grad, node)
        return sum_grad / size_diff

def mean(a, axes=None, keepdims=False):
    return Mean(axes, keepdims)(a)

# class Var(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
#         self.axes = axes
#         self.keepdims = keepdims

#     def compute(self, a):
#         if self.axes is not None:
#             return array_api.var(a, axis=self.axes, keepdims=self.keepdims)
#         else:
#             return array_api.var(a, keepdims=self.keepdims)

#     def gradient(self, out_grad, node):
#         X = node.inputs[0]

#         X_mean = X.mean(axes=self.axes, keepdims=self.keepdims).broadcast_to(X.shape)

#         y = ((X - X_mean)**2).mean(axes=self.axes, keepdims=self.keepdims)
#         y.backward()

#         return out_grad * X.grad

# def var(a, axes=None, keepdims=False):
#     return Var(axes, keepdims)(a)

# class LogSumExp(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         if axes is None:
#             self.axes = None
#         else:
#             self.axes = axes if isinstance(axes, tuple) else (axes, )

#     @staticmethod
#     def get_output_shape(inp_shape: List[int], axes: Optional[tuple] = None):
#         outp_shape = inp_shape.copy()

#         if axes is None:
#             outp_shape[:] = [1] * len(inp_shape)
#         else:
#             for ax in axes:
#                 outp_shape[ax] = 1

#         return outp_shape

#     def compute(self, Z):
#         m = array_api.max(Z, axis=self.axes, keepdims=True)
#         return array_api.log(array_api.exp((Z - m)).sum(axis=self.axes)) + array_api.max(Z, axis=self.axes)

#     def max(self, Z, dims=True):
#         return Tensor(array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=dims), device=Z.device)

#     def gradient(self, out_grad, node):
#         Z = node.inputs[0]
#         inp_shape = list(Z.shape)
#         out_shape = LogSumExp.get_output_shape(inp_shape, axes=self.axes)
#         m = self.max(Z)
#         print("Z: \n", Z)
#         print("m: \n", broadcast_to(m, Z.shape))
#         d = EWiseAdd()(Z, -broadcast_to(m, Z.shape))
#         print("Z-m: \n", d)
#         y = log(d).sum(axes=self.axes) + self.max(Z, False)
#         print(y)
#         y.backward()

#         inp_shape = tuple(node.inputs[0].shape)

#         if self.axes is not None:
#             out_grad = broadcast_to(reshape(out_grad, out_shape), inp_shape)
#         else:
#             out_grad = broadcast_to(reshape(out_grad, out_shape), inp_shape)

#         return out_grad * Z.grad

def get_unsq_outp_shape(inp_shape: List[int], axes: Optional[tuple] = None):
        outp_shape = inp_shape.copy()

        if axes is None:
            outp_shape[:] = [1] * len(inp_shape)
        else:
            for ax in axes:
                outp_shape[ax] = 1

        return outp_shape

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes if isinstance(axes, tuple) else (axes, )
 
    def broadcasted_max(self, inp):
        # analog of np.max(arr, self.axes, keepdims=True)
        unsq_outp_shape = get_unsq_outp_shape(list(inp.shape), self.axes)
 
        is_tensor = isinstance(inp, Tensor)
        
        inp_data = inp
        try:
            inp.realize_cached_data()
            inp_data = inp.cached_data
        except:
            pass
 
        inp_max = inp_data.max(self.axes)
        broadcasted_max = inp_max.reshape(unsq_outp_shape)
        broadcasted_max = array_api.broadcast_to(broadcasted_max, inp.shape)
 
        return Tensor(broadcasted_max, device=inp.device) if is_tensor \
            else broadcasted_max
 
    def compute(self, Z: numpy.ndarray):
        max_z = Z.max(self.axes)
        broadcasted_max_z = self.broadcasted_max(Z)
 
        return array_api.log(
                    array_api.sum(
                        array_api.exp(Z - broadcasted_max_z), 
                        self.axes
                    )
                ) + max_z
 
 
    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        inp_shape = list(inp.shape)
        unsq_outp_shape = get_unsq_outp_shape(inp_shape, self.axes)
 
        broadcasted_max = self.broadcasted_max(inp)
 
        # Calculating numerically stable Softmax
        inp -= broadcasted_max
        exp_inp = exp(inp)
        sum_exp = exp_inp.sum(self.axes).reshape(unsq_outp_shape)
        sum_exp = sum_exp.broadcast_to(inp_shape)
        
        softmax = EWiseDiv()(exp_inp, sum_exp)
 
        broadcasted_out_grad = broadcast_to(out_grad.reshape(unsq_outp_shape), 
                                            inp_shape)
        return softmax * broadcasted_out_grad


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class ILostAnyHope(TensorOp):
    def compute(self, a, b):
        print(type(a), type(Tensor(b)))
        return a + Tensor(b, dtype=b.dtype)