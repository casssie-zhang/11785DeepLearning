import numpy as np

from mytorch import tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return (tensor.Tensor(grad_output.data.T),)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = tensor.Tensor(grad_output.data * np.exp(a.data))
        grad_a = unbroadcast(grad_a, a.shape)
        return grad_a

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data


        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not type(a).__name__ == 'Tensor':
            a = tensor.Tensor(a)
        if not type(b).__name__ == 'Tensor':
            b = tensor.Tensor(b)


        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = (-1) * np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b



class Mul(Function):
    """element wise multiplication"""
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}, ".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = b.data * np.ones(a.shape) * grad_output.data
        grad_b = a.data * np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class MatMul(Function):
    """matrix multiplication"""
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}, ".format(type(a).__name__, type(b).__name__))

        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.dot(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.dot(grad_output.data, b.data.T)
        grad_b = np.dot(a.data.T, grad_output.data)

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).

                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        ctx.save_for_backward(x)
        ctx.is_train = is_train
        ctx.p = p
        requires_grad = x.requires_grad

        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")

        if is_train:
            mask = np.random.binomial(np.ones(x.shape).astype(int), 1-p)
            masked_x = mask * x.data / (1 - p)
            out = tensor.Tensor(masked_x, requires_grad=requires_grad, is_leaf=not requires_grad)
        else:
            mask = np.ones(x.shape).astype(int)
            out = tensor.Tensor(x.data, requires_grad = requires_grad, is_leaf=not requires_grad)

        ctx.mask = mask

        return out
        # raise NotImplementedError("TODO: Implement Dropout(Function).forward() for hw1 bonus!")

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.mask
        p = ctx.p
        if ctx.is_train:
            grad = grad_output.data * mask / (1 - p)
        else:
            grad = grad_output.data

        grad = tensor.Tensor(grad)
        return grad


        # raise NotImplementedError("TODO: Implement Dropout(Function).backward() for hw1 bonus!")


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):

        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.divide(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.divide(1, b.data) * np.ones(a.shape) * grad_output.data
        grad_b = (-1) * np.divide(a.data, b.data * b.data) * np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b:int):
        ctx.save_for_backward(a)
        ctx.up = b
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data ** b, requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        b = ctx.up
        grad_a = tensor.Tensor(grad_output.data * b *(a.data ** (b-1)))

        return grad_a, None



class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(a.data, 0), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data * (a.data > 0).astype(int))


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1 - b)
        return tensor.Tensor(grad)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1 - out ** 2)
        return tensor.Tensor(grad)


class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv2d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size, output_size) output data
        """
        ctx.stride = stride
        ctx.save_for_backward(x, weight, bias)

        batch_size, in_channel, input_size, _ = x.shape
        out_channel, _, kernel_size, _ = weight.shape

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size, output_size))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for w in range(output_size):
            for h in range(output_size):
                for j in range(out_channel):
                    #slice shape  = (batch_size, in_channel, kernel_size, kernel_size)
                    segment = x.data[:, :, stride * w : stride * w+kernel_size, stride * h : stride * h + kernel_size]
                    assert segment.shape == (batch_size, in_channel, kernel_size, kernel_size)
                    out[:, j, w, h] = np.sum(np.multiply(weight.data[j,:,:,:], segment),axis=(1,2,3)) + bias.data[j]


        requires_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
        # TODO: Put output into tensor with correct settings and return
        out_tensor = tensor.Tensor(out, requires_grad=requires_grad)
        out_tensor.is_leaf = not out_tensor.requires_grad
        return out_tensor
        # raise NotImplementedError("Implement functional.Conv1d.forward()!")

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        out_channel, in_channel, kernel_size, _ = weight.shape
        batch_size, out_channel, output_size, _ = grad_output.shape
        batch_size, in_channel, input_size, _ = x.shape


        # calculate gradient w.r.t X (y in l-1 layer)
        # upsampling
        if stride > 1:
            grad_up = np.zeros((batch_size, out_channel, input_size-kernel_size+1, input_size-kernel_size+1))
            for i in range(output_size):
                for j in range(output_size):
                    grad_up[:, :, i * stride, j * stride] = grad_output.data[:, :, i, j]
        else:
            grad_up = grad_output.data

        grad_pad = np.pad(grad_up, ((0,0), (0,0), (kernel_size-1, kernel_size-1), (kernel_size-1, kernel_size-1)))
        assert grad_pad.shape == (batch_size, out_channel, input_size + kernel_size - 1, input_size + kernel_size - 1)

        # flip weight along with the kernel axis
        flip_weight = np.flip(weight.data, axis=(2,3))
        # print(weight, flip_weight)

        # conv grad,
        grad_x = np.zeros(x.shape)
        grad_w = np.zeros(weight.shape)
        grad_bias = grad_output.data.sum(axis=(0,2,3))

        for j in range(in_channel):
            for w in range(input_size):
                for h in range(input_size):
                    segment = grad_pad[:, :, w : w + kernel_size, h : h + kernel_size]
                    assert segment.shape == (batch_size, out_channel, kernel_size, kernel_size)
                    # shape1 = (out_channel, kernel_size, kernel_size)
                    # shape2 = (batch_size, out_channel, kernel_size, kernel_size)
                    grad_x[:, j, w, h] = np.sum(np.multiply(flip_weight[:, j, :, :], segment), axis=(1,2,3))

        grad_filter_size = grad_up.shape[-1]
        for k in range(out_channel):
            for w_w in range(kernel_size):
                for w_h in range(kernel_size):
                    for j in range(in_channel):
                        # grad_up[:, k, :] shape: (batch_size, 1, input_size-kernel+1)
                        # x.data[:, j, w:w+input_size-kernel_size + 1] batch size, , input_size-kernel-size+1
                        grad_w[k, j, w_w, w_h] = np.multiply(grad_up[:, k, :],
                                                             x.data[:,j, w_w:w_w+grad_filter_size, w_h:w_h+grad_filter_size]).sum()

        # delete pad zeros
        grad_x = tensor.Tensor(grad_x)
        assert grad_x.shape == x.shape
        grad_w = tensor.Tensor(grad_w)
        assert grad_w.shape == weight.shape
        grad_bias = tensor.Tensor(grad_bias)
        assert bias.shape == grad_bias.shape

        return grad_x, grad_w, grad_bias

class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape

        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias, tensor.Tensor(stride))

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for i in range(output_size):
            for j in range(out_channel):
                #slice shape  = (batch_size, in_channel, kernel_size)
                segment = x.data[:, :, stride * i : stride * i+kernel_size]
                assert segment.shape == (batch_size, in_channel, kernel_size)
                out[:, j, i] = np.sum(np.multiply(weight.data[j,:,:], segment),axis=(1,2)) + bias.data[j]


        requires_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
        # TODO: Put output into tensor with correct settings and return
        out_tensor = tensor.Tensor(out, requires_grad=requires_grad)
        out_tensor.is_leaf = not out_tensor.requires_grad
        return out_tensor
        # raise NotImplementedError("Implement functional.Conv1d.forward()!")

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        x, weight, bias, stride = ctx.saved_tensors
        stride = stride.data
        out_channel, in_channel, kernel_size = weight.shape
        batch_size, out_channel, output_size = grad_output.shape
        batch_size, in_channel, input_size = x.shape


        # calculate gradient w.r.t X (y in l-1 layer)
        # upsampling
        if stride > 1:
            grad_up = np.zeros((batch_size, out_channel, input_size-kernel_size+1))
            for i in range(output_size):
                grad_up[:, :, i * stride] = grad_output.data[:, :, i]
        else:
            grad_up = grad_output.data

        grad_pad = np.pad(grad_up, ((0,0), (0,0), (kernel_size-1, kernel_size-1)))
        assert grad_pad.shape == (batch_size, out_channel, input_size + kernel_size - 1)

        # flip weight along with the kernel axis
        flip_weight = np.flip(weight.data, axis=2)

        # conv grad,
        grad_x = np.zeros(x.shape)
        grad_w = np.zeros(weight.shape)
        grad_bias = grad_output.data.sum(axis=(0,2))

        for j in range(in_channel):
            for i in range(input_size):
                segment = grad_pad[:, :, i : i + kernel_size]
                assert segment.shape == (batch_size, out_channel, kernel_size)
                # shape1 = (out_channel, kernel_size)
                # shape2 = (batch_size, out_channel, kernel_size)
                grad_x[:, j, i] = np.sum(np.multiply(flip_weight[:, j, :], segment), axis=(1,2))

        grad_filter_size = grad_up.shape[-1]
        for k in range(out_channel):
            for w in range(kernel_size):
                for j in range(in_channel):
                    # grad_up[:, k, :] shape: (batch_size, 1, input_size-kernel+1)
                    # x.data[:, j, w:w+input_size-kernel_size + 1] batch size, , input_size-kernel-size+1
                    grad_w[k, j, w] = np.multiply(grad_up[:, k, :], x.data[:,j, w:w+grad_filter_size]).sum()

        # delete pad zeros
        grad_x = tensor.Tensor(grad_x)
        assert grad_x.shape == x.shape
        grad_w = tensor.Tensor(grad_w)
        assert grad_w.shape == weight.shape
        grad_bias = tensor.Tensor(grad_bias)
        assert bias.shape == grad_bias.shape

        return grad_x, grad_w, grad_bias, None



def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.

        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return (input_size - kernel_size) // stride + 1
    # raise NotImplementedError("TODO: Complete functional.get_conv1d_output_size()!")


def log_softmax(logits):
    """applies logsoftmax to the logits, using LogSumExp Trick
     Args:
         logits(Tensor):
     Returns:
         Tensor: log soft max
         """

    # max_x = tensor.Tensor(np.max(logits.data, axis=1).reshape(-1, 1), requires_grad=False)
    max_x = tensor.Tensor.zeros(logits.data.shape[0], 1)
    sum_exp = (logits - max_x).exp().sum(axis=1, keepdims=True)
    log_sum_exp = sum_exp.log()

    return logits - (max_x + log_sum_exp)

def nll_loss(logsoftmax, target, batch_size):
    return tensor.Tensor(-1) * (logsoftmax * target).sum() / tensor.Tensor(batch_size)


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape
    labels = to_one_hot(target, num_classes)
    return nll_loss(log_softmax(predicted), labels, batch_size=batch_size)




    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.


    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    raise Exception("TODO: Implement XELoss for comp graph")

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)


class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        ctx.indices = indices
        ctx.save_for_backward(x)
        requires_grad = x.requires_grad

        return tensor.Tensor(x.data[indices], requires_grad=requires_grad,
                             is_leaf = not requires_grad)
        # raise NotImplementedError('Implemented Slice.forward')

    @staticmethod
    def backward(ctx,grad_output):
        x = ctx.saved_tensors[0]
        indices = ctx.indices

        grad_x = np.zeros(x.shape)
        grad_x[indices] = grad_output.data
        return grad_x, None

        # raise NotImplementedError('Implemented Slice.backward')

class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            dim (int): The dimension along which we concatenate our tensors
            seq (list of tensors): list of tensors we wish to concatenate
        '''
        *seq, dim = args
        ctx.dim = dim
        ctx.save_for_backward(*seq)

        seq_data = [t.data for t in seq]
        concat_array = np.concatenate(seq_data, axis=dim)
        requires_grad = any([t.requires_grad for t in seq])
        concat_tensor = tensor.Tensor(concat_array, requires_grad=requires_grad)
        concat_tensor.is_leaf = concat_tensor.requires_grad

        return concat_tensor
        # raise NotImplementedError('Implement Cat.forward')

    @staticmethod
    def backward(ctx,grad_output):
        seq = ctx.saved_tensors
        dim = ctx.dim
        gradients = grad_output.data

        split_idx = np.cumsum([t.data.shape[dim] for t in seq])
        seq_grads = np.split(gradients, split_idx[:-1], axis=dim)


        assert len(seq_grads) == len(seq)

        seq_grads = [tensor.Tensor(grad) for grad in seq_grads]
        return tuple(seq_grads + [None])


# def argmax_nd(arr, dim):


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a max over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batch_size, in_channel, input_size, _ = x.shape
        if not stride:
            stride = 1
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        out_channel = in_channel
        out = np.zeros((batch_size, out_channel, output_size, output_size))
        out_idx = np.zeros((batch_size, out_channel, output_size, output_size))

        for w in range(output_size):
            for h in range(output_size):
                for j in range(out_channel):
                    segment = x.data[:, j, w*stride:w*stride + kernel_size, h*stride:h*stride+kernel_size]
                    assert (segment.shape == (batch_size, kernel_size, kernel_size))
                    # pidx = [np.unravel_index(np.argmax(r), r.shape) for r in segment]
                    pidx = [np.argmax(r) for r in segment]
                    out[:, j, w, h] = np.max(segment, axis=(1,2))
                    out_idx[:, j, w, h] = pidx

                    # out[:, j, w, h] = np.

        requires_grad = x.requires_grad
        ctx.pidx = out_idx
        # TODO: Put output into tensor with correct settings and return
        out_tensor = tensor.Tensor(out, requires_grad=requires_grad)
        out_tensor.is_leaf = not out_tensor.requires_grad
        return out_tensor


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        # raise Exception("TODO: Finish MaxPool2d(Function).backward() for hw2 bonus")
        x = ctx.saved_tensors[0]
        batch_size, in_channel, input_size, _ = x.shape
        _, _, output_size, _ = grad_output.shape
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        out_idx = ctx.pidx

        grad_x = np.zeros(x.shape)

        for j in range(in_channel):
            for w in range(output_size):
                for h in range(output_size):
                    window_idx = out_idx[:, j, w, h]
                    window_idx = [np.unravel_index(int(i), (kernel_size, kernel_size)) for i in window_idx]\
                                 + np.array([w*stride, h*stride])
                    grad_x[:, j, window_idx[:,0], window_idx[:,1]] += grad_output.data[:, j, w, h].reshape(-1,1)


        grad_x = tensor.Tensor(grad_x)

        return grad_x





class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a mean over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batch_size, in_channel, input_size, _ = x.shape
        if not stride:
            stride = 1
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        out_channel = in_channel
        out = np.zeros((batch_size, out_channel, output_size, output_size))
        # out_idx = np.zeros((batch_size, out_channel, output_size, output_size))

        for w in range(output_size):
            for h in range(output_size):
                for j in range(out_channel):
                    segment = x.data[:, j, w*stride:w*stride + kernel_size, h*stride:h*stride+kernel_size]
                    assert (segment.shape == (batch_size, kernel_size, kernel_size))
                    # pidx = [np.unravel_index(np.argmax(r), r.shape) for r in segment]
                    # pidx = [np.argmax(r) for r in segment]
                    out[:, j, w, h] = np.mean(segment, axis=(1,2))
                    # out_idx[:, j, w, h] = pidx

                    # out[:, j, w, h] = np.

        requires_grad = x.requires_grad
        # ctx.pidx = out_idx
        # TODO: Put output into tensor with correct settings and return
        out_tensor = tensor.Tensor(out, requires_grad=requires_grad)
        out_tensor.is_leaf = not out_tensor.requires_grad
        return out_tensor


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        # raise Exception("TODO: Finish AvgPool2d(Function).backward() for hw2 bonus")
        # raise Exception("TODO: Finish MaxPool2d(Function).backward() for hw2 bonus")
        x = ctx.saved_tensors[0]
        batch_size, in_channel, input_size, _ = x.shape
        _, _, output_size, _ = grad_output.shape
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        # out_idx = ctx.pidx

        grad_x = np.zeros(x.shape)

        for j in range(in_channel):
            for w in range(output_size):
                for h in range(output_size):
                    grad_x[:, j, w * stride:w * stride + kernel_size, h * stride:h * stride + kernel_size] \
                        += (grad_output.data[:, j, w, h] / (kernel_size * kernel_size)).reshape(-1,1,1)


        grad_x = tensor.Tensor(grad_x)

        return grad_x

