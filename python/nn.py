import loop_tool_py as lt

const_map = {}


def fill(constant, symbolic_shape):
    if constant in const_map:
        const = const_map[constant]
    else:
        const = lt.Tensor().set(constant)
        const_map[constant] = const
    return const


def mean(X, dims):
    exprs = [(x, lt.Expr(0)) for x in dims]
    one = fill(1, dims).to(*dims, constraints=exprs)
    return X.sum(*dims) / one.sum(*dims)


def sigmoid(T):
    shape = T.symbolic_shape
    one = fill(1, shape)
    return (one + (-T).exp()).reciprocal()


def swish(T):
    shape = T.symbolic_shape
    return T * sigmoid(T)


def relu(T):
    shape = T.symbolic_shape
    zero = fill(0, shape)
    return T.max(zero)


def relu6(T):
    shape = T.symbolic_shape
    six = fill(6, shape)
    return relu(T) - relu(T - six)


def hardswish(T):
    shape = T.symbolic_shape
    three = fill(3, shape)
    sixth = fill(1 / 6, shape)
    return T * relu6(T + three) * sixth


def tanh(T):
    shape = T.symbolic_shape
    two = fill(2, shape)
    one = fill(1, shape)
    return two * sigmoid(two * T) - one


def linear(X, W, bias=None):
    reduction_dims = set(X.symbolic_shape) & set(W.symbolic_shape)
    Y = (X * W).sum(*reduction_dims)
    if bias:
        Y = Y + bias
    return Y


# pad(X, (s.K, 1), (s.X, (0, 1)))
def pad(X, *args):
    for d, pad in args:
        if type(pad) is tuple:
            X = X.pad(d, *pad)
        else:
            X = X.pad(d, pad)
    return X


def conv(X, W, spatial, window, stride=1, channel_reduce=True):
    assert len(spatial) == len(window)
    # output dimensions need new names
    new_spatial = [lt.Symbol(x.name + "o") for x in spatial]
    outer = [d for d in X.symbolic_shape if d not in spatial]
    exprs = [lt.Expr(stride) * x + k for x, k in zip(new_spatial, window)]
    X = X.to(*outer, *new_spatial, *window, constraints=zip(spatial, exprs))

    # reduce over input channels and the windowed dims
    if channel_reduce:
        reduction_dims = (set(X.symbolic_shape) & set(W.symbolic_shape)) | set(window)
    else:
        reduction_dims = set(window)
    return (X * W).sum(*reduction_dims)


def batch_norm(x, mean, var, weight, bias, eps=lt.Tensor().set(1e-5)):
    x = (x - mean) * weight
    return x / (var + eps).sqrt() + bias
