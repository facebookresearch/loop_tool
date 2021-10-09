import loop_tool_py as lt
import numpy as np
import time

L = 1024
if "cuda" in lt.backends():
    lt.set_default_hardware("cuda")
    lt.set_default_backend("cuda")
    L *= 1024

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


def split(loop, parallel_size, inner_size):
    assert loop.tail == 0
    s = loop.size // (parallel_size * inner_size)
    t = loop.size % (parallel_size * inner_size)
    return [
        (loop.var, (s, t)),
        (loop.var, (parallel_size, 0)),
        (loop.var, (inner_size, 0)),
    ]


loop_tree = Z.loop_tree
ir = loop_tree.ir

for l in loop_tree.loops:
    if loop_tree.trivially_parallel(l):
        loop = loop_tree.loop(l)
        for n in ir.nodes:
            ir.set_order(n, split(loop, 128, 4))
            ir.disable_reuse(n, 2)

loop_tree = lt.LoopTree(ir)
# parallelize the outermost loops
loop_tree.annotate(loop_tree.loops[0], "parallel")
loop_tree.annotate(loop_tree.loops[1], "parallel")

Z.set(loop_tree)

print(Z.loop_tree)
bench(loop_tree, 10, 1000)
