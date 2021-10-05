import loop_tool_py as lt
import numpy as np

m, n, k = 8, 8, 8
A = lt.Tensor(m, k)
B = lt.Tensor(k, n)
A.set(np.random.randn(m, k))
B.set(np.random.randn(k, n))


def mm(A, B):
    N = lt.Symbol("N")
    M = lt.Symbol("M")
    K = lt.Symbol("K")
    C = A.to(M, K) * B.to(K, N)
    return C.sum(K)


C = mm(A, B)
C_ref = A.numpy() @ B.numpy()
assert np.allclose(C.numpy(), C_ref, atol=0.0001, rtol=0.0001)


def conv(X, W):
    Ni = lt.Symbol("Ni")
    No = lt.Symbol("No")
    K = lt.Symbol("K")
    # im2col
    X = X.to(Ni).to(No, K, constraints=[(Ni, No + K)])
    # just a matmul
    Y = (X * W.to(K)).sum(K)
    return Y


X = lt.Tensor(128)
W = lt.Tensor(3)
X.set(np.random.randn(128))
W.set(np.random.randn(3))

Y = conv(X, W)
Y_ref = np.correlate(X.numpy(), W.numpy(), mode="valid")

assert np.allclose(Y.numpy(), Y_ref, atol=0.0001, rtol=0.0001)

# unbound sizes can be inferred

N = lt.Symbol("N")
K = lt.Symbol("K")

X = lt.Tensor(N)
W0 = lt.Tensor(3)
W1 = lt.Tensor(8)

Y = conv(X, W)
Z = (Y.to(K) * W1.to(K)).sum()

Z.unify()
assert X.shape[0] == 10
