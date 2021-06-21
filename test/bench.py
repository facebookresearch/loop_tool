import loop_tool_py as lt
import random
import numpy as np
import math
import time


def gen_pw_add():
    ir = lt.IR()
    a = ir.create_var("a")
    r0 = ir.create_node("read", [], [a])
    r1 = ir.create_node("read", [], [a])
    add = ir.create_node("add", [r0, r1], [a])
    w = ir.create_node("write", [add], [a])
    ir.set_inputs([r0, r1])
    ir.set_outputs([w])
    return ir, a


def test_pw(size, inner_size, vec_size):
    assert size >= (inner_size * vec_size)
    ir, v = gen_pw_add() # v = pointwise var
    size_map = {}
    size_map[v] = size
    for n in ir.nodes:
      outer = size // (inner_size * vec_size)
      outer_rem = size % (inner_size * vec_size)

      ir.set_order(n, [
        (v, (outer, outer_rem)),
        (v, (inner_size, 0)),
        (v, (vec_size, 0))
        ])
      ir.disable_reuse(n, 2)
    loop_tree = lt.LoopTree(ir)
    A = lt.Tensor(size)
    B = lt.Tensor(size)
    C = lt.Tensor(size)
    Ap = np.random.randn(size)
    Bp = np.random.randn(size)
    A.set(Ap)
    B.set(Bp)
    C_ref = Ap + Bp
    C.set(1337.0)
    parallel = set(loop_tree.children(loop_tree.roots[0]))
    c = lt.CompiledCuda(loop_tree, parallel)
    c([A, B, C])
    C_test = C.to_numpy()
    max_diff = np.max(np.abs(C_test - C_ref))
    mean_val = np.mean(np.abs(C_ref))
    assert max_diff < 1e-3 * mean_val
    iters = 10000
    # warmup
    for i in range(50):
      c([A, B, C])
    t = time.time()
    for i in range(iters - 1):
      c([A, B, C], False)
    c([A, B, C])
    t_ = time.time()
    #print(loop_tree.dump(lambda x: "[threaded]" if x in parallel else ""))
    #print(c.code)
    # 2 read 1 write, 4 bytes per float
    bytes_moved = (2 + 1) * 4 * size * iters / (t_ - t) / 1e9
    pct = bytes_moved / c.bandwidth
    usec = (t_ - t) / iters * 1e6
    #print(f"peak: {c.bandwidth} GB/sec")
    print(f"{bytes_moved:.2f} GB/sec", f"({100 * pct:.2f}% of peak, {usec:.2f} usec per iter)")
    return bytes_moved, c.code, loop_tree.dump(lambda x: "// Threaded" if x in parallel else "")

s = 1024 * 1024
best = 0
code = ""
loop_tree = ""
for i in range(1, 8):
  inner = 512 * 16 * i
  for vec_pow in range(3):
    vec = 2 ** vec_pow
    b, c, l = test_pw(s, inner, vec)
    if b > best:
      best = b
      code = c
      loop_tree = l
print(f"Best kernel found ({best:.2f} GB/sec):")
print(loop_tree)
print(code)

