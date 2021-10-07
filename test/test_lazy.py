import loop_tool_py as lt
import numpy as np
import time

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
    K = lt.Symbol("Kc")
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

Y = conv(X, W0)
Z = (Y.to(K) * W1.to(K)).sum(K)

Z.unify()
assert X.shape[0] == 10
Z.compile()

# we can override the schedule
def schedule(ir):
  #print(ir)
  return ir

Z.set(schedule(Z.ir))
print(Z.loop_tree)

W0.set(np.ones(3))
W1.set(np.ones(8))
X.set(np.ones(10))

assert Z.numpy() == 24

lt.set_default_hardware("cuda")
lt.set_default_backend("cuda")

L = 1024 * 32

X = lt.Tensor(L)
Y = lt.Tensor(L)
X.set(np.random.randn(L))
Y.set(np.random.randn(L))

N = lt.Symbol("N")
Z = X.to(N) + Y.to(N)

assert np.allclose(Z.numpy(), X.numpy() + Y.numpy(), atol=0.0001, rtol=0.0001)

def bench(loop_tree, warmup, iters):
  X = lt.Tensor(L)
  Y = lt.Tensor(L)
  X.set(np.random.randn(L))
  Y.set(np.random.randn(L))
  N = lt.Symbol("N")
  Z = X.to(N) + Y.to(N)
  Z.set(loop_tree)

  for i in range(warmup):
    Z = X.to(N) + Y.to(N)
    Z.resolve()
  t1 = time.time()
  for i in range(iters):
    Z = X.to(N) + Y.to(N)
    Z.resolve()
  t2 = time.time()
  print(f"{iters / (t2 - t1):.2f} iters/sec")

print(Z.loop_tree)
bench(Z.loop_tree, 10, 100)

loop_tree = Z.loop_tree
for l in loop_tree.loops:
  loop_tree.annotate(l, "parallel")
print(loop_tree)
bench(loop_tree, 10, 100)
