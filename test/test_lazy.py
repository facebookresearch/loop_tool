import loop_tool_py as lt
import numpy as np
import time


backend = "cpu"
if "cuda" in lt.backends():
    backend = "cuda"
    lt.set_default_hardware("cuda")
    lt.set_default_backend("cuda")


m, n, k = 8, 8, 8
A = lt.Tensor(m, k).set(np.random.randn(m, k))
B = lt.Tensor(k, n).set(np.random.randn(k, n))


def mm(A, B):
    N = lt.Symbol("N")
    M = lt.Symbol("M")
    K = lt.Symbol("K")
    C = A.to(M, K) * B.to(K, N)
    return C.sum(K)


C = mm(A, B)
if backend == "cuda":
    print(C.compiled.code)
C_ref = A.numpy() @ B.numpy()
assert np.allclose(C.numpy(), C_ref, atol=0.0001, rtol=0.0001)


def conv(X, W):
    with lt.SymbolGenerator() as s:
        return (X[s.No + s.K] * W.to(s.K)).sum(s.K)


X = lt.Tensor(128).set(np.random.randn(128))
W = lt.Tensor(3).set(np.random.randn(3))

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
    # print(ir)
    return ir


Z.set(schedule(Z.ir))
print(Z.loop_tree)

W0.set(np.ones(3))
W1.set(np.ones(8))
X.set(np.ones(10))

assert Z.numpy() == 24

L = 1024 * 128

X = lt.Tensor(L)
Y = lt.Tensor(L)
X.set(np.random.randn(L))
Y.set(np.random.randn(L))

N = lt.Symbol("N")
Z = X.to(N) + Y.to(N)
print(Z.loop_tree)

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
bench(Z.loop_tree, 10, 1000)


def split(loop, inner_size):
    assert loop.tail == 0
    s = loop.size // inner_size
    t = loop.size % inner_size
    return [(loop.var, (s, t)), (loop.var, (inner_size, 0))]


loop_tree = Z.loop_tree
ir = loop_tree.ir

for l in loop_tree.loops:
    if loop_tree.trivially_parallel(l):
        loop = loop_tree.loop(l)
        for n in ir.nodes:
            ir.set_order(n, split(loop, L // 2))

loop_tree = lt.LoopTree(ir)
# parallelize the outermost loop
loop_tree.annotate(loop_tree.loops[0], "parallel")

Z.set(loop_tree)

print(Z.loop_tree)
bench(loop_tree, 10, 1000)
