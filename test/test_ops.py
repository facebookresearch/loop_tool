import loop_tool_py as lt
import nn
import numpy as np
import tinygrad
from tinygrad import nn as tg_nn
import tinygrad.tensor as tg


def test_exp():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)

    B_np = np.exp(A_np)
    B_lt = A_lt.exp().numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_recip():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)

    B_np = 1 / A_np
    B_lt = A_lt.reciprocal().numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_add_one():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)

    B_np = 1 + A_np
    B_lt = (lt.Tensor().set(1) + A_lt).numpy()
    assert np.allclose(B_np, B_lt, rtol=0.001, atol=0.001)


def test_mean():
    N = 32
    M = 32
    A_np = np.random.randn(M, N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.mean(0).data
    B_lt = nn.mean(A_lt, [A_lt.symbolic_shape[0]]).numpy()
    assert np.allclose(B_tg, B_lt, rtol=0.001, atol=0.001)


def test_sigmoid():
    N = 32
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.sigmoid().data
    B_lt = nn.sigmoid(A_lt).numpy()
    assert np.allclose(B_tg, B_lt, rtol=0.001, atol=0.001)


def test_swish():
    N = 128
    A_np = np.random.randn(1, 32, 8, 8)
    A_tg = tg.Tensor(A_np)
    A_lt = lt.Tensor(A_np)

    B_tg = A_tg.swish().data
    B_lt = nn.swish(A_lt).numpy()
    assert np.allclose(B_tg, B_lt, rtol=0.001, atol=0.001)


def test_relu():
    N = 23
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.relu().data
    B_lt = nn.relu(A_lt)


def test_relu6():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.relu6().data
    B_lt = nn.relu6(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_hardswish():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.hardswish().data
    B_lt = nn.hardswish(A_lt)
    assert np.allclose(B_tg, B_lt.numpy(), rtol=0.01, atol=0.01)


def test_tanh():
    N = 128
    A_np = np.random.randn(N)
    A_lt = lt.Tensor(A_np)
    A_tg = tg.Tensor(A_np)

    B_tg = A_tg.tanh().data
    B_lt = nn.tanh(A_lt)

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
    A_lt = lt.Tensor(s.M, s.K).set(A_np)
    B_lt = lt.Tensor(s.K, s.N).set(B_np)
    bias_lt = lt.Tensor(s.N).set(bias_np)

    C_tg = A_tg.linear(B_tg, bias_tg)
    C_lt = nn.linear(A_lt, B_lt, bias_lt)

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
    X_lt = lt.Tensor(s.b, s.ic, s.y, s.x).set(X)
    W_lt = lt.Tensor(s.oc, s.ic, s.wy, s.wx).set(W)

    Y_tg = X_tg.conv2d(W_tg)
    Y_lt = nn.conv(X_lt, W_lt, [s.y, s.x], [s.wy, s.wx])

    # transpose to compare with the hardcoded layout of tinygrad
    b, y, x, c = Y_lt.symbolic_shape
    Y_lt = Y_lt.transpose(b, c, y, x)

    assert np.allclose(Y_tg.data, Y_lt.numpy(), rtol=0.01, atol=0.01)


def test_batch_norm():
    B = 8
    C = 16 // 2
    Y = 32 // 4
    X = 32 // 4

    s = lt.SymbolGenerator()
    A_np = np.random.randn(B, C, Y, X)
    A_tg = tg.Tensor(A_np)
    A_lt = lt.Tensor(s.B, s.C, s.Y, s.X).set(A_np)

    bn = tg_nn.BatchNorm2D(C)
    bn.running_mean.data = np.random.randn(C)
    bn.running_var.data = np.abs(np.random.randn(C))
    bn.weight.data = np.random.randn(C)
    bn.bias.data = np.random.randn(C)

    B_tg = bn(A_tg)

    mean = lt.Tensor(s.C).set(bn.running_mean.data)
    var = lt.Tensor(s.C).set(bn.running_var.data)
    weight = lt.Tensor(s.C).set(bn.weight.data)
    bias = lt.Tensor(s.C).set(bn.bias.data)

    B_lt  = nn.batch_norm(A_lt, mean, var, weight, bias)

    assert np.allclose(B_tg.data, B_lt.numpy(), rtol=0.001, atol=0.001)

def test_pad():
    N = 8 // 8
    M = 8 // 8
    A_np = np.random.randn(1, 1, M, N)
    A_tg = tg.Tensor(A_np)
    A_lt = lt.Tensor(A_np)
    B_tg = A_tg.pad2d(padding=[2,3,2,3])
    _, _, m, n = A_lt.symbolic_shape
    B_lt = nn.pad(A_lt, (m, (2, 3)), (n, (2, 3)))

    assert np.allclose(B_tg.data, B_lt.numpy(), rtol=0.001, atol=0.001)

def test_pad_pad():
    A_np = np.random.randn(1,1)
    A_lt = lt.Tensor(lt.Symbol("Y"), lt.Symbol("X")).set(A_np)
    X = A_lt.symbolic_shape[1]
    A_lt = nn.pad(A_lt, (X, 1))
    print("pad 0", A_lt.shape)
    X = A_lt.symbolic_shape[0]
    A_lt = nn.pad(A_lt, (X, 1))
    print("pad 1", A_lt.shape)
    X = A_lt.symbolic_shape[0]
    A_lt = nn.pad(A_lt, (X, 1))
    print("pad 2", A_lt.shape)
    #print(A_lt.ir)
    print(A_lt.loop_tree)
    #print(A_lt.code)
    print(A_lt.shape)
    print(A_lt.numpy())

def test_pad_conv():
    C = 1
    N = 3#64 // 8
    M = 3#64 // 8
    K = 3
    A_np = np.random.randn(1, C, M, N)
    W_np = np.random.randn(1, C, K, K)
    A_np = np.ones((1, C, M, N))
    W_np = np.ones((1, C, K, K))

    A_tg = tg.Tensor(A_np)
    W_tg = tg.Tensor(W_np)
    s = lt.SymbolGenerator()
    A_lt = lt.Tensor(s.b, s.c, s.y, s.x).set(A_np)
    W_lt = lt.Tensor(s.co, s.c, s.ky, s.kx).set(W_np)

    B_tg = A_tg.pad2d(padding=[2,3,2,3])
    _, c, m, n = A_lt.symbolic_shape
    B_lt = nn.pad(A_lt, (m, (2,3)), (n, (2,3)))
    assert np.allclose(B_tg.data, B_lt.numpy(), rtol=0.001, atol=0.001)
    print("good")

    C_tg = B_tg.conv2d(W_tg)
    _, c, m, n = B_lt.symbolic_shape
    _, c, km, kn = W_lt.symbolic_shape
    C_lt = nn.conv(B_lt, W_lt, [m, n], [km, kn])
    b, y, x, c = C_lt.symbolic_shape
    C_lt = C_lt.transpose(b, c, y, x)
    print(C_lt.code)

    print(C_tg.data.shape, C_lt.numpy().shape)
    print(C_tg.data, C_lt.numpy())

    assert np.allclose(C_tg.data, C_lt.numpy(), rtol=0.001, atol=0.001)


#test_exp()
#test_recip()
#test_add_one()
#test_mean()
#test_sigmoid()
#test_swish()
#test_relu()
#test_relu6()
#test_hardswish()
#test_tanh()
#test_linear()
#test_conv()
#test_batch_norm()
#test_pad()
test_pad_pad()
#test_pad_conv()
