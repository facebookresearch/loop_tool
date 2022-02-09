# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import loop_tool as lt
import random
import numpy as np
import math
import time

# this works with both CPU and CudaGPU backends
lt.set_default_hardware("cuda")


def gen_pw_add():
    ir = lt.IR()
    a = ir.create_var("a")
    r0 = ir.create_node(lt.read, [], [a])
    r1 = ir.create_node(lt.read, [], [a])
    add = ir.create_node(lt.add, [r0, r1], [a])
    w = ir.create_node(lt.write, [add], [a])
    ir.set_inputs([r0, r1])
    ir.set_outputs([w])
    return ir, a


def gen_reduce_add():
    ir = lt.IR()
    n = ir.create_var("N")
    r = ir.create_node(lt.read, [], [n])
    add = ir.create_node(lt.add, [r], [])
    w = ir.create_node(lt.write, [add], [])
    ir.set_inputs([r])
    ir.set_outputs([w])
    return ir, n


def gen_mm():
    ir = lt.IR()
    m = ir.create_var("m")
    n = ir.create_var("n")
    k = ir.create_var("k")
    r0 = ir.create_node(lt.read, [], [m, k])
    r1 = ir.create_node(lt.read, [], [k, n])
    mul = ir.create_node(lt.mul, [r0, r1], [m, n, k])
    add = ir.create_node(lt.add, [mul], [m, n])
    w = ir.create_node(lt.write, [add], [m, n])
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
        vs = ir.loop_vars(n)
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

    def comp_val(val, backend, c=None):
        val = val.to_numpy()
        max_diff = np.max(np.abs(val.flatten() - ref.flatten()))
        max_idx = np.argmax(np.abs(val.flatten() - ref.flatten()))
        mean_val = np.mean(np.abs(ref))
        debug_info = f"{loop_tree.dump(l)}"
        if backend == "cuda":
            debug_info += f"\n{c.code}\n{val.flatten()}\n{ref.flatten()}\n"  # {np.abs(val.flatten() - ref.flatten())}"
        assert (
            max_diff < 1e-3 * mean_val
        ), f"diff is {max_diff} at {max_idx} ({val.flatten()[max_idx]} vs {ref.flatten()[max_idx]}) on {backend} (mean is {mean_val}) for:\n{debug_info}"

    cpu_fn = lt.cpu(loop_tree)
    cpu_fn(inps)
    comp_val(inps[-1], "cpu")

    if "cuda" in lt.backends():
        c = lt.cuda(loop_tree, set(cuda_threads))
        c(inps)
        comp_val(inps[-1], "cuda", c)


def test_rand_pw(size):
    ir, v = gen_pw_add()
    size_map = {}
    size_map[v] = size
    ir = rand_sched(ir, size_map)
    loop_tree = lt.LoopTree(ir)
    A = lt.RawTensor(size)
    B = lt.RawTensor(size)
    C = lt.RawTensor(size)
    Ap = np.random.randn(size)
    Bp = np.random.randn(size)
    A.set(Ap)
    B.set(Bp)
    C_ref = Ap + Bp
    C.set(1337.0)
    p = set(loop_tree.roots)
    l = lambda x: "[thread]" if x in p else ""
    try:
        c = lt.cuda(loop_tree, p)
    except Exception as e:
        print(loop_tree.dump(l))
        raise
    try:
        c([A, B, C])
    except Exception as e:
        print(loop_tree.dump(l))
        print(c.code)
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
    A = lt.RawTensor(M * K)
    B = lt.RawTensor(K * N)
    C = lt.RawTensor(M * N)
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
    A = lt.RawTensor(size)
    B = lt.RawTensor(1)
    Ap = np.random.randn(size)  # np.random.randn(size)
    A.set(Ap)
    B_ref = np.sum(Ap)
    B.set(0.0)
    # print(loop_tree.to_cuda(set()))
    p = set([l for l in loop_tree.loops if loop_tree.trivially_parallel(l)])
    check_exec(loop_tree, [A, B], B_ref, p)


def test_annot(N, M):
    ir = lt.IR()
    n = ir.create_var("N")
    m = ir.create_var("M")
    r = ir.create_node(lt.read, [], [n, m])
    add0 = ir.create_node(lt.add, [r], [n])
    add1 = ir.create_node(lt.add, [add0], [])
    w = ir.create_node(lt.write, [add1], [])
    ir.set_inputs([r])
    ir.set_outputs([w])

    # 1. pick node (discrete named space, dynamic on program)
    # 2. choose type of thing (discrete named space)
    #   3. set order <-- (list of choice spaces (either named discrete vairables or int64range)
    #   3. disable reuse (discrete)
    #   3. thread (discrete)

    # ir.set_order(r, [n : Var, 8, 0, m : Var, 7, 0])
    # for n in 8:
    #  for m in 7:
    #    r[n * 7 + m]
    # ir.set_order(r, [n : Var, 2, 2, m : Var, 7, 0, n : Var, 3, 0])
    # for n in 2 r 2:
    #  for m in 7:
    #    for n_1 in 3:
    #      r[(n * 3 + n_1) * 7 + m]

    # ===
    # [1, 2, 4, 5, 6]
    # [1, 2, (4, 5, 6)]
    # [1, 2, (5, 4, 6)]
    # [1, (2, 5, 4), 6]
    # [1, (5, 2, 4), 6]
    # [(1, 5, 2), 4, 6]

    # for n in 2:
    #  for m in 7:
    #    for n_1 in 3:
    #      r[(n * 3 + n_1) * 7 + m]
    # n = 2
    #  for m in 7:
    #    for n_1 in 2:
    #      r[(n * 3 + n_1) * 7 + m]

    # ir.disable_reuse(r)
    # ir.thread(r)

    ir.set_order(add0, [(n, (N, 0)), (m, (M, 0))])
    ir.disable_reuse(r, m)
    ir.disable_reuse(add0, n)
    ir.set_order(add1, [(n, (N, 0))])
    ir.set_order(w, [])
    loop_tree = lt.LoopTree(ir)
    loop_tree.annotate(loop_tree.roots[0], "parallel")
    print(loop_tree)
    cpu_fn = lt.cpu(loop_tree)
    A = lt.RawTensor(N * M)
    B = lt.RawTensor(1)
    Ap = np.random.randn(N * M)
    print(lt.backends())

    # A = lt.tensor("K", "J")
    # read = lt.tensor("J")
    # A_tmp = lt.tensor("K")
    # B = lt.tensor()
    # for k in lt.iter("K", 16, backend="parallel"):
    #  for j in lt.iter("J", 16):
    #    read[j] = A[k, j]
    #  for j in lt.iter("J", 16):
    #    A_tmp[k] += read[j]
    # for k in lt.iter("K", 16):
    #  B[] += A_tmp[k]

    A.set(Ap)
    B_ref = np.sum(Ap)
    B.set(0.0)
    cpu_fn([A, B])
    t = time.time()
    for i in range(100):
        cpu_fn([A, B])
    print(time.time() - t)
    print(B.to_numpy(), B_ref)


if __name__ == "__main__":
    random.seed(1337)
    np.random.seed(1337)
    test_annot(32, 1024)
    print("pointwise", end="")
    # powers of 2
    sizes = [2 ** i for i in range(1, 10)]
    # odd numbers near powers of 2
    sizes += [2 ** i + 1 for i in range(1, 8)]
    # primes
    sizes += [x for x in range(8, 30) if all(x % y != 0 for y in range(2, x))]
    for s in sizes:
        print(".", end="", flush=True)
        for _ in range(5):
            test_rand_pw(s)
    print("pass!")
    print("reduce", end="")
    for s in sizes:
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
