import torch
import numpy as np

from mytorch.tensor import Tensor
from mytorch.autograd_engine import *
from mytorch.nn.functional import *
from mytorch.nn.conv import Conv1d
from mytorch.nn.batchnorm import BatchNorm1d
import torch.nn as nn

"""Use this file to help you develop operations/functions.
It actually works fairly similarly to the autograder.
We've provided many test functions.
For your own operations, implement tests for them here to easily
debug your code."""

def main():
    """Runs test methods in order shown below."""
    # test four basic ops
    # test_sum()
    # test_logsoftmax()
    # test_exp()
    # test_matmul()
    # test_add()
    # test_sub()
    # test_mul()
    # test_div()

    # you probably want to verify
    # any other ops you create...


    # test autograd
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test_constant()
    # test7()
    # test8()
    #
    # # for when you might want it...
    # testbroadcast()
    # test_conv_forward_back_just_1_layer()


    test_bn()


def test_bn():
    x = Tensor(np.array([[19., 33.], [18., 35.], [18., 35.]]))
    x.requires_grad = True
    bn1 = BatchNorm1d(2)
    bn1.is_train = True
    out = bn1(x)
    out.sum().backward()
    # out.backward()
    gamma_grad = np.array([7.54951657e-15, -7.54951657e-15])
    beta_grad = np.array([3., 3.])
    running_mean = np.array([1.83333333, 3.43333333])
    running_var = np.array([0.93333333, 1.03333333])
    outout = np.array([[ 1.41418174, -1.41420561],
                       [-0.70709087,  0.7071028 ],
                       [-0.70709087,  0.7071028 ]])
    x_grad = np.array([[-7.53627424e-15, -3.76820070e-15],
                        [ 3.76813712e-15,  1.88410035e-15],
                        [ 3.76813712e-15,  1.88410035e-15]])
    def check(a, b, eps=1e-8):
        max_dif = np.abs(a - b).max()
        if (max_dif < eps): return True
        else: return False
    assert(check(gamma_grad, bn1.gamma.grad.data))
    assert(check(beta_grad, bn1.beta.grad.data))
    assert(check(running_mean, bn1.running_mean.data))
    assert(check(running_var, bn1.running_var.data))
    assert(check(outout, out.data))
    print(x_grad)
    print(x.grad.data)
    assert(check(x_grad, x.grad.data))

def test_conv_forward_back_just_1_layer():
    in_c = 2
    out_c = 2
    kernel = 2
    stride = 1
    width = 5
    batch_size = 1

    # setup weights and biases
    conv = Conv1d(in_c, out_c, kernel, stride)
    conv.weight = Tensor(np.random.randint(5, size=conv.weight.shape)+.0,
                requires_grad = True)
    conv.bias = Tensor(np.random.randint(2, size=conv.out_channel)+.0,
                requires_grad = True)
    conv_torch = nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride)
    conv_torch.weight = nn.Parameter(torch.tensor(conv.weight.data))
    conv_torch.bias = nn.Parameter(torch.tensor(conv.bias.data))
    print(f"weight:\n {conv.weight}")
    print(f"bias:\n {conv.bias}")

    # setup input
    x = Tensor(np.random.randint(5, size=(batch_size, in_c, width)),\
                requires_grad = True)
    x_torch = get_same_torch_tensor(x).double()
    print(f"x:\n {x}")

    # calculate output
    o = conv(x)
    o_torch = conv_torch(x_torch)
    print(f"out:\n {o}")

    # backward
    o.backward()
    o_torch.sum().backward()
    print(f"grad_x:\n {x.grad}")
    print(f"grad_w:\n {conv.weight.grad}")
    print(f"grad_b:\n {conv.bias.grad}")

    # check everything
    assert check_val_and_grad(x, x_torch)
    assert check_val_and_grad(o, o_torch)
    assert check_val_and_grad(conv.weight, conv_torch.weight)
    assert check_val_and_grad(conv.bias, conv_torch.bias)

def test_sum():
    a = Tensor.randn(3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = a.sum()
    b_torch = a_torch.sum()

    b.backward()
    b_torch.backward()

    check_val_and_grad(a, a_torch)


def test_logsoftmax():
    a = Tensor.randn(3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = log_softmax(a)
    b_torch = a_torch.log_softmax(dim = 1)

    b.backward()
    b_torch.sum().backward()

    check_val_and_grad(b, b_torch)
    check_val_and_grad(a, a_torch)


def test_exp():
    a = Tensor.randn(3,4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = a.exp()
    b_torch = a_torch.exp()
    b.backward()
    b_torch.sum().backward()

    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(a, a_torch)



def test_matmul():
    """Tests that mytorch's elementwise multiplication matches torch's"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(4, 5)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a * b'
    ctx = ContextManager()
    c = MatMul.forward(ctx, a, b)
    c_torch = a_torch @ b_torch

    # run mytorch and torch multiplication backward
    back = MatMul.backward(ctx, Tensor.ones(3,5))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure * is overridden
    c_using_override = a @ b
    assert check_val(c_using_override, c_torch)

    return True



def test_add():
    """Tests that mytorch addition matches torch's addition"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a + b'
    ctx = ContextManager()
    c = Add.forward(ctx, a, b)
    c_torch = a_torch + b_torch

    # run mytorch and torch addition backward
    back = Add.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val_and_grad(back[0], a_torch.grad)
    assert check_val_and_grad(back[1], b_torch.grad)

    # ensure + is overridden
    c_using_override = a + b
    assert check_val(c_using_override, c_torch)

    return True

def test_sub():
    """Tests that mytorch subtraction matches torch's subtraction"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a - b'
    ctx = ContextManager()
    c = Sub.forward(ctx, a, b)
    c_torch = a_torch - b_torch

    # run mytorch and torch subtraction backward
    back = Sub.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure - is overridden
    c_using_override = a - b
    assert check_val(c_using_override, c_torch)

    return True

def test_mul():
    """Tests that mytorch's elementwise multiplication matches torch's"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a * b'
    ctx = ContextManager()
    c = Mul.forward(ctx, a, b)
    c_torch = a_torch * b_torch

    # run mytorch and torch multiplication backward
    back = Mul.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure * is overridden
    c_using_override = a * b
    assert check_val(c_using_override, c_torch)

    return True


def test_div():
    """Tests that mytorch division matches torch's"""

    # shape of tensor to test
    shape = (1, 2, 3)

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(*shape)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(*shape)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a / b'
    ctx = ContextManager()
    c = Div.forward(ctx, a, b)
    c_torch = a_torch / b_torch

    # run mytorch and torch division backward
    back = Div.backward(ctx, Tensor.ones(*shape))
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val(back[0], a_torch.grad)
    assert check_val(back[1], b_torch.grad)

    # ensure / is overridden
    c_using_override = a / b
    assert check_val(c_using_override, c_torch)

    return True


def testbroadcast():
    """Tests addition WITH broadcasting matches torch's"""

    # shape of tensor to test

    # get mytorch and torch tensor: 'a'
    a = Tensor.randn(3, 4)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    # get mytorch and torch tensor: 'b'
    b = Tensor.randn(4)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    # run mytorch and torch forward: 'c = a + b'
    c = a + b
    c_torch = a_torch + b_torch

    # run mytorch and torch addition backward
    c.backward()
    c_torch.sum().backward()

    # check that c matches
    assert check_val_and_grad(c, c_torch)
    # check that dc/da and dc/db respectively match
    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)


# addition, requires grad
def test1():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a + b
    c_torch = a_torch + b_torch

    c_torch.sum().backward()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)

# multiplication, requires grad
def test2():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a * b
    c_torch = a_torch * b_torch

    c_torch.sum().backward()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)

# addition, one arg requires grad
def test3():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = False
    b_torch = get_same_torch_tensor(b)

    c = a + b
    c_torch = a_torch + b_torch

    c_torch.sum().backward()
    c.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)

# the example from writeup
def test4():
    a = Tensor(1, requires_grad = True)
    a_torch = get_same_torch_tensor(a)

    b = Tensor(2, requires_grad = True)
    b_torch = get_same_torch_tensor(b)

    c = Tensor(3, requires_grad = True)
    c_torch = get_same_torch_tensor(c)

    d = a + a * b
    d_torch = a_torch + a_torch * b_torch

    e = d + c + Tensor(3)
    e_torch = d_torch + c_torch + torch.tensor(3)

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)


# the example from writeup, more strict
def test5():
    a = Tensor(1, requires_grad = True)
    a_torch = get_same_torch_tensor(a)

    b = Tensor(2, requires_grad = True)
    b_torch = get_same_torch_tensor(b)

    c = Tensor(3, requires_grad = True)
    c_torch = get_same_torch_tensor(c)

    # d = a + a * b
    z1 = a * b
    z1_torch = a_torch * b_torch
    d = a + z1
    d_torch = a_torch + z1_torch

    # e = (d + c) + 3
    z2 = d + c
    z2_torch = d_torch + c_torch
    e = z2 + Tensor(3)
    e_torch = z2_torch + 3

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(z1, z1_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(z2, z2_torch)
    assert check_val_and_grad(e, e_torch)


# more complicated tests
def test6():
    a = Tensor.randn(2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a / b
    c_torch = a_torch / b_torch

    d = a - b
    d_torch = a_torch - b_torch

    e = c + d
    e_torch = c_torch + d_torch

    e.backward()
    e_torch.sum().backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)
    assert check_val_and_grad(e, e_torch)

def test_constant():
    # c = 5
    c = Tensor(5., requires_grad=True)
    c_torch = get_same_torch_tensor(c)
    z2 = Tensor(3) * c
    z2_torch = 3 * c_torch

    z2.backward()
    z2_torch.backward()

    assert check_val_and_grad(c, c_torch)



# another fun test
def test7():
    # a = 3
    a = Tensor(3., requires_grad=False)
    a_torch = get_same_torch_tensor(a)

    # b = 4
    b = Tensor(4., requires_grad=False)
    b_torch = get_same_torch_tensor(b)

    # c = 5
    c = Tensor(5., requires_grad=True)
    c_torch = get_same_torch_tensor(c)

    # out = a * b + 3 * c
    z1 = a * b
    z1_torch = a_torch * b_torch
    z2 = Tensor(3) * c
    z2_torch = 3 * c_torch
    out = z1 + z2
    out_torch = z1_torch + z2_torch

    out_torch.sum().backward()
    out.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(z1, z1_torch)
    assert check_val_and_grad(z2, z2_torch)
    assert check_val_and_grad(out, out_torch)

# non-tensor arguments
def test8():
    a = Tensor.randn(1, 2, 3)
    a.requires_grad = True
    a_torch = get_same_torch_tensor(a)

    b = Tensor.randn(1, 2, 3)
    b.requires_grad = True
    b_torch = get_same_torch_tensor(b)

    c = a + b
    c_torch = a_torch + b_torch

    d = c.reshape(-1)
    d_torch = c_torch.reshape(-1)

    d_torch.sum().backward()
    d.backward()

    assert check_val_and_grad(a, a_torch)
    assert check_val_and_grad(b, b_torch)
    assert check_val_and_grad(c, c_torch)
    assert check_val_and_grad(d, d_torch)


"""General-use helper functions"""

def get_same_torch_tensor(mytorch_tensor):
    """Returns a torch tensor with the same data/params as some mytorch tensor"""
    res = torch.tensor(mytorch_tensor.data).double()
    res.requires_grad = mytorch_tensor.requires_grad
    return res


def check_val_and_grad(mytorch_tensor, pytorch_tensor):
    """Compares values and params of mytorch and torch tensors.
    
    Returns:
        boolean: False if not similar, True if similar"""
    return check_val(mytorch_tensor, pytorch_tensor) and \
           check_grad(mytorch_tensor, pytorch_tensor)


def check_val(mytorch_tensor, pytorch_tensor, eps=1e-10):
    """Compares the data values of mytorch/torch tensors."""
    if not isinstance(pytorch_tensor, torch.DoubleTensor):
        print("Warning: torch tensor is not a DoubleTensor. It is instead {}".format(pytorch_tensor.type()))
        print("It is highly recommended that similarity testing is done with DoubleTensors as numpy arrays have 64-bit precision (like DoubleTensors)")

    if tuple(mytorch_tensor.shape) != tuple(pytorch_tensor.shape):
        print("mytorch tensor and pytorch tensor has different shapes: {}, {}".format(
            mytorch_tensor.shape, pytorch_tensor.shape
        ))
        return False

    data_diff = np.abs(mytorch_tensor.data - pytorch_tensor.data.numpy())
    max_diff = data_diff.max()
    if max_diff < eps:
        return True
    else:
        print("Data element differs by {}:".format(max_diff))
        print("mytorch tensor:")
        print(mytorch_tensor)
        print("pytorch tensor:")
        print(pytorch_tensor)

        return False

def check_grad(mytorch_tensor, pytorch_tensor, eps = 1e-10):
    """Compares the gradient of mytorch and torch tensors"""
    if mytorch_tensor.grad is None or pytorch_tensor_nograd(pytorch_tensor):
        if mytorch_tensor.grad is None and pytorch_tensor_nograd(pytorch_tensor):
            return True
        elif mytorch_tensor.grad is None:
            print("Mytorch grad is None, but pytorch is not")
            return False
        else:
            print("Pytorch grad is None, but mytorch is not")
            return False

    grad_diff = np.abs(mytorch_tensor.grad.data - pytorch_tensor.grad.data.numpy())
    max_diff = grad_diff.max()
    if max_diff < eps:
        return True
    else:
        print("Grad differs by {}".format(grad_diff))
        return False

def pytorch_tensor_nograd(pytorch_tensor):
    return not pytorch_tensor.requires_grad or not pytorch_tensor.is_leaf

if __name__ == "__main__":
    main()
