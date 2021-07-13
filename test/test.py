# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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


def gen_reduce_add():
    ir = lt.IR()
    n = ir.create_var("N")
    r = ir.create_node("read", [], [n])
    add = ir.create_node("add", [r], [])
    w = ir.create_node("write", [add], [])
    ir.set_inputs([r])
    ir.set_outputs([w])
    return ir, n


def gen_mm():
    ir = lt.IR()
    m = ir.create_var("m")
    n = ir.create_var("n")
    k = ir.create_var("k")
    r0 = ir.create_node("read", [], [m, k])
    r1 = ir.create_node("read", [], [k, n])
    mul = ir.create_node("mul", [r0, r1], [m, n, k])
    add = ir.create_node("add", [mul], [m, n])
    w = ir.create_node("write", [add], [m, n])
    ir.set_inputs([r0, r1])
    ir.set_outputs([w])
    return ir, m, n, k


def get_total_size(splits):
    running = 1
    for split in splits[::-1]:
        running = split[0] * running + split[1]
    return running


def do_split(splits, idx, new_size):
    inner = splits[idx + 1 :]
    inner_size = get_total_size(inner)
    assert new_size[1] < inner_size
    outer = splits[idx]
    outer_size = outer[0] * inner_size + outer[1]
    new_total = new_size[0] * inner_size + new_size[1]
    new_outer = (outer_size // new_total, outer_size % new_total)
    new_splits = splits[:idx] + [new_outer, new_size] + inner
    assert get_total_size(new_splits) == get_total_size(splits)
    return new_splits


def rand_split(splits, attempts):
    for _ in range(attempts):
        idx = random.randint(0, len(splits) - 1)
        try:
            s = random.randint(2, int(math.sqrt(splits[idx][0])))
        except:
            continue
        splits = do_split(splits, idx, (s, 0))
    return splits


def rand_sched(ir, size_map):
    # generate a random schedule
    for n in ir.nodes:
        vs = ir.all_vars(n)
        splits = {}
        for v in vs:
            v_splits = rand_split([(size_map[v], 0)], random.randint(1, 3))
            assert get_total_size(v_splits) == size_map[v]
            splits[v] = zip([v for _ in v_splits], v_splits)
        order = [s for _, vs in splits.items() for s in vs]
        ir.set_order(n, order)
    return ir


def check_exec(loop_tree, inps, ref, cuda_threads=[]):

    l = lambda x: "[thread]" if x in cuda_threads else ""

    def comp_val(val, backend):
        val = val.to_numpy()
        max_diff = np.max(np.abs(val.flatten() - ref.flatten()))
        mean_val = np.mean(np.abs(ref))
        assert (
            max_diff < 1e-3 * mean_val
        ), f"diff is {max_diff} on {backend} (mean is {mean_val}) for:\n{loop_tree.dump(l)}"

    loop_tree(inps)
    comp_val(inps[-1], "cpu")

    if "cuda" in lt.backends():
        c = lt.CompiledCuda(loop_tree, set(cuda_threads))
        c(inps)
        comp_val(inps[-1], "cuda")


def test_rand_pw(size):
    ir, v = gen_pw_add()
    size_map = {}
    size_map[v] = size
    ir = rand_sched(ir, size_map)
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
    p = set(loop_tree.roots)
    l = lambda x: "[thread]" if x in p else ""
    try:
        c = lt.CompiledCuda(loop_tree, p)
        c([A, B, C])
    except Exception as e:
        print(loop_tree.dump(l))
        raise
    check_exec(loop_tree, [A, B, C], C_ref, p)


def test_rand_mm(M, N, K):
    ir, m, n, k = gen_mm()
    size_map = {}
    size_map[m] = M
    size_map[n] = N
    size_map[k] = K
    ir = rand_sched(ir, size_map)
    loop_tree = lt.LoopTree(ir)
    A = lt.Tensor(M * K)
    B = lt.Tensor(K * N)
    C = lt.Tensor(M * N)
    Ap = np.random.randn(M, K)
    Bp = np.random.randn(K, N)
    Ap = np.ones((M, K))
    # Bp = np.ones((K, N))
    Ap = np.arange(M * K).reshape((M, K))
    Bp = np.arange(K * N).reshape((K, N))
    A.set(Ap)
    B.set(Bp)
    C_ref = Ap @ Bp
    C.set(1337.0)
    # loop_tree.exec_cpu([A, B, C])
    p = [l for l in loop_tree.loops if loop_tree.trivially_parallel(l)]
    p = loop_tree.roots
    check_exec(loop_tree, [A, B, C], C_ref, p)


def test_rand_reduce(size):
    ir, v = gen_reduce_add()
    size_map = {}
    size_map[v] = size
    ir = rand_sched(ir, size_map)
    loop_tree = lt.LoopTree(ir)
    A = lt.Tensor(size)
    B = lt.Tensor(1)
    Ap = np.random.randn(size)  # np.random.randn(size)
    A.set(Ap)
    B_ref = np.sum(Ap)
    B.set(0.0)
    # print(loop_tree.to_cuda(set()))
    p = set([l for l in loop_tree.loops if loop_tree.trivially_parallel(l)])
    check_exec(loop_tree, [A, B], B_ref, p)


if __name__ == "__main__":
    random.seed(1337)
    np.random.seed(1337)
    print("pointwise", end="")
    for s in [7, 64, 128, 129, 512]:
        print(".", end="", flush=True)
        for _ in range(5):
            test_rand_pw(s)
    print("pass!")
    print("reduce", end="")
    for s in [7, 64, 128, 129, 512]:
        print(".", end="", flush=True)
        for _ in range(5):
            test_rand_reduce(s)
    print("pass!")
    print("mm", end="")
    for m in range(1, 17, 3):
        for n in range(5, 7):
            for k in range(8, 11):
                print(".", end="", flush=True)
                for _ in range(2):
                    test_rand_mm(m, n, k)
    print("pass!")
