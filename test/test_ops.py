import loop_tool_py as lt
import numpy as np
import tinygrad
import tinygrad.tensor as tg

const_map = {}


def fill(constant, symbolic_shape):
    if constant in const_map:
        const = const_map[constant]
    else:
        const = lt.Tensor(1).set(constant)
        const_map[constant] = const
    k = const.symbolic_shape[0]
    return const.to(*symbolic_shape, constraints=[(k, lt.Expr(0))])


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


def conv(X, W, spatial, window):
    assert len(spatial) == len(window)
    # output dimensions need new names
    new_spatial = [lt.Symbol(x.name + "o") for x in spatial]
    outer = [d for d in X.symbolic_shape if d not in spatial]
    exprs = [x + k for x, k in zip(new_spatial, window)]
    X = X.to(*outer, *new_spatial, *window, constraints=zip(spatial, exprs))

    # reduce over input channels and the windowed dims
    reduction_dims = (set(X.symbolic_shape) & set(W.symbolic_shape)) | set(window)
    return (X * W).sum(*reduction_dims)


def test_exp():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)

    B_np = np.exp(A_np)
    B_lt = A_lt.exp().numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_recip():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)

    B_np = 1 / A_np
    B_lt = A_lt.reciprocal().numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_add_one():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)

    B_np = 1 + A_np
    B_lt = (lt.Tensor(1).set(1) + A_lt).numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_sigmoid():
    N = 32
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.sigmoid().data
    B_lt = sigmoid(A_lt).numpy()
    assert np.allclose(B_tg, B_lt, rtol=0.001, atol=0.001)


def test_swish():
    N = 4
    A_np = np.random.randn(N)
    A_np = np.ones((N,))
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.swish().data
    B_lt = swish(A_lt).numpy()
    assert np.allclose(B_tg, B_lt, rtol=0.01, atol=0.01)


def test_relu():
    N = 23
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.relu().data
    B_lt = relu(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_relu6():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.relu6().data
    B_lt = relu6(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_hardswish():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.hardswish().data
    B_lt = hardswish(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_tanh():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(N).set(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.tanh().data
    B_lt = tanh(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_linear():
    M = 123
    N = 231
    K = 128
    A_np = np.random.randn(M, K)
    B_np = np.random.randn(K, N)
    bias_np = np.random.randn(N)

    A_tg = tg.Tensor(A_np)
    B_tg = tg.Tensor(B_np)
    bias_tg = tg.Tensor(bias_np)

    s = lt.SymbolGenerator()
    A_lt = lt.Tensor(s.M, s.K).set_size(M, K).set(A_np)
    B_lt = lt.Tensor(s.K, s.N).set_size(K, N).set(B_np)
    bias_lt = lt.Tensor(s.N).set_size(N).set(bias_np)

    C_tg = A_tg.linear(B_tg, bias_tg)
    C_lt = linear(A_lt, B_lt, bias_lt)

    assert np.allclose(C_tg.data, C_lt.numpy(), rtol=0.01, atol=0.01)


def test_conv():
    batch = 33
    spatial = 44
    window = 3
    in_channel = 16
    out_channel = 8
    X = np.random.randn(batch, in_channel, spatial, spatial)
    W = np.random.randn(out_channel, in_channel, window, window)

    X_tg = tg.Tensor(X)
    W_tg = tg.Tensor(W)

    s = lt.SymbolGenerator()
    X_lt = (
        lt.Tensor(s.b, s.ic, s.y, s.x)
        .set_size(batch, in_channel, spatial, spatial)
        .set(X)
    )
    W_lt = (
        lt.Tensor(s.oc, s.ic, s.wy, s.wx)
        .set_size(out_channel, in_channel, window, window)
        .set(W)
    )

    Y_tg = X_tg.conv2d(W_tg)
    Y_lt = conv(X_lt, W_lt, [s.y, s.x], [s.wy, s.wx])

    # transpose to compare with the hardcoded layout of tinygrad
    b, y, x, c = Y_lt.symbolic_shape
    Y_lt = Y_lt.transpose(b, c, y, x)

    assert np.allclose(Y_tg.data, Y_lt.numpy(), rtol=0.01, atol=0.01)


test_exp()
test_recip()
test_add_one()
test_sigmoid()
test_swish()
test_relu()
test_relu6()
test_hardswish()
test_tanh()
test_linear()
test_conv()
