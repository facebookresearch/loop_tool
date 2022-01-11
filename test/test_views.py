import loop_tool_py as lt
import numpy as np


def test_pad():
    m, n = 128, 16

    base_np = np.random.randn(m, n)
    padded_np = np.pad(base_np, [(0,), (3,)])

    base_lt = lt.Tensor(m, n).set(base_np)
    padded_lt = base_lt.pad(base_lt.symbolic_shape[1], 3)

    assert np.allclose(padded_lt.numpy(), padded_np)


def test_concat():
    m, n, k = 128, 16, 5

    A_np = np.random.randn(m, n)
    B_np = np.random.randn(m, k)
    C_np = np.concatenate((A_np, B_np), axis=1)

    A_lt = lt.Tensor(m, n).set(A_np)
    B_lt = lt.Tensor(m, k).set(B_np)

    with lt.SymbolGenerator() as s:
        C_lt = A_lt.to(s.m, s.n) | B_lt.to(s.m, s.k)

    assert np.allclose(C_lt.numpy(), C_np)


def test_2d_conv():
    import torch
    import torch.nn.functional as F

    def conv2d(X, W):
        s = lt.SymbolGenerator()
        X = X[s.C, s.H + s.Kh, s.W + s.Kw]
        W = W.to(s.Co, s.C, s.Kh, s.Kw)
        return (X * W).sum(s.C, s.Kh, s.Kw).transpose(s.Co, s.H, s.W)

    ci = 16
    co = 16
    x = 8
    k = 3
    X_np = np.random.randn(ci, x, x)
    W_np = np.random.randn(co, ci, k, k)
    Y_np = F.conv2d(torch.tensor(X_np).unsqueeze(0), torch.tensor(W_np)).numpy()

    X_lt = lt.Tensor(ci, x, x).set(X_np)
    W_lt = lt.Tensor(co, ci, k, k).set(W_np)
    Y_lt = conv2d(X_lt, W_lt)

    assert np.allclose(Y_lt.numpy(), Y_np, rtol=0.001, atol=0.001)


def test_padded_2d_conv():
    import torch
    import torch.nn.functional as F

    def conv2d(X, W):
        s = lt.SymbolGenerator()
        X = X.to(s.c, s.h, s.w).pad(s.h, 1).pad(s.w, 1)
        X = X[s.C, s.H + s.Kh, s.W + s.Kw]
        W = W.to(s.Co, s.C, s.Kh, s.Kw)
        return (X * W).sum(s.C, s.Kh, s.Kw).transpose(s.Co, s.H, s.W)

    ci = 16
    co = 16
    x = 8
    k = 3
    X_np = np.random.randn(ci, x, x)
    W_np = np.random.randn(co, ci, k, k)
    Y_np = F.conv2d(
        torch.tensor(X_np).unsqueeze(0), torch.tensor(W_np), padding=1
    ).numpy()

    X_lt = lt.Tensor(ci, x, x).set(X_np)
    W_lt = lt.Tensor(co, ci, k, k).set(W_np)
    Y_lt = conv2d(X_lt, W_lt)
    print(Y_lt.loop_tree)

    assert np.allclose(Y_lt.numpy(), Y_np, rtol=0.001, atol=0.001)


def test_many_pad():
    import string

    N = 5
    a = lt.Tensor(lt.Symbol("A")).set(np.random.randn(N))
    X = a.symbolic_shape[0]
    Y = lt.Symbol("B")
    b = a.to(
        Y, constraints=[(Y, X + lt.Expr(1)), (lt.Size(Y), lt.Size(X) + lt.Expr(2))]
    )
    for i in range(10):
        X = b.symbolic_shape[0]
        Y = lt.Symbol(string.ascii_uppercase[i + 2])
        b = b.to(
            Y, constraints=[(Y, X + lt.Expr(1)), (lt.Size(Y), lt.Size(X) + lt.Expr(2))]
        )
    ir = b.ir
    vs = ir.vars
    print(vs)
    for n in ir.nodes:
        if "write" in ir.dump(n):
            print(ir.dump(n))
            ir.set_order(n, [(vs[0], (N, 0))])
    # b.set(ir)
    print(b.loop_tree)
    print(b.code)
    print(b.numpy())


def test_many_conv():
    import string

    N = 11
    # a = lt.Tensor(lt.Symbol("A")).set(np.arange(N))
    a = lt.Tensor(lt.Symbol("A")).set(np.ones(N))
    X = a.symbolic_shape[0]
    Y = lt.Symbol("B")
    Z = lt.Symbol("C")
    cur_syms = [Y, Z]
    b = a.to(*cur_syms, constraints=[(X, Y + Z), (lt.Size(Z), lt.Expr(3))])
    for i in range(2):
        X = b.symbolic_shape[0]
        Y = lt.Symbol(string.ascii_uppercase[i * 2 + 3])
        Z = lt.Symbol(string.ascii_uppercase[i * 2 + 4])
        cur_syms = [Y, Z] + cur_syms[1:]
        # print("setting size of ", Y, Z, b.shape)
        b = b.to(*cur_syms, constraints=[(X, Y + Z), (lt.Size(Z), lt.Expr(3))])
    print(b.ir)
    print(b.loop_tree)
    print(b.shape)
    print(b.numpy())
    print(b.code)


# test_pad()
# test_concat()
# test_2d_conv()
# test_padded_2d_conv()
test_many_pad()
# test_many_conv()
