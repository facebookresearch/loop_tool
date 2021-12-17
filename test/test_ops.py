import loop_tool_py as lt
import numpy as np
import tinygrad
import tinygrad.tensor as tg


def fill(constant, symbolic_shape):
    s = lt.SymbolGenerator()
    const = lt.Tensor(1).set(constant)
    k = const.symbolic_shape[0]
    return const.to(*symbolic_shape, constraints=[(k, lt.Expr(0))])


def sigmoid(T):
    N = T.symbolic_shape[0]
    one = fill(1, [N])
    return (one + (-T).exp()).reciprocal()


def swish(T):
    N = T.symbolic_shape[0]
    return T * sigmoid(T)


def relu(T):
    zero = fill(0, T.symbolic_shape)
    return T.max(zero)


def relu6(T):
    N = T.symbolic_shape[0]
    six = fill(6, [N])
    return relu(T) - relu(T - six)


def hardswish(T):
    N = T.symbolic_shape[0]
    three = fill(3, [N])
    sixth = fill(1 / 6, [N])
    return T * relu6(T + three) * sixth


def tanh(T):
    N = T.symbolic_shape[0]
    two = fill(2, [N])
    one = fill(1, [N])
    return two * sigmoid(two * T) - one


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
    N = 128
    A_np = np.random.randn(N)
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


test_exp()
test_recip()
test_add_one()
test_sigmoid()
test_swish()
test_relu()
test_relu6()
test_hardswish()
test_tanh()
