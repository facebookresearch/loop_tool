import loop_tool as lt
import numpy as np

M = 128 * 2
N = 128 * 2
K = 128 * 2

A_np = np.random.randn(M, K)
B_np = np.random.randn(K, N)

def mm(a, b):
    m, n, k = lt.symbols("m n k")
    return (a.to(m, k) * b.to(k, n)).sum(k)

A = lt.Tensor(A_np)
B = lt.Tensor(B_np)

with lt.Backend("loop_nest"):
    C = mm(A, B)
    lt.ui(C)
    tree = C.loop_tree
 
    #tree = tree.annotate(tree.loops[0], "[loop_nest]")
    #print(hash(C))
    #print(C.numpy())
    #print(C.numpy())
    C.set(tree)
    print(C.code)
    #C = mm(A, B)
    #C.clear_cache()
    #print(C.numpy())
