# `loop_tool`

`loop_tool` is a lightweight, highly-portable IR for compiling N-dimensional data computation.

## Features

1. Lightweight dense linear algebra IR
   1. Cheaply copied/cloned
   2. Core lib < 400KB when compiled in Release mode
   3. Simplistic semantics (nothing more than a pure functional dataflow graph)
2. IR manipulation
   1. DFG annotation (annotate compute nodes with loop order + loop share-ability)
   2. Loop tree annotation for CUDA threading (simply pass in a set of loop to thread)
   
## Tutorial

Import `loop_tool` and define an IR

```python
import loop_tool_py as lt

ir = lt.IR()

a = ir.create_var("a")
# arguments: type of node, inputs, variables
r0 = ir.create_node("read", [], [a])
r1 = ir.create_node("read", [], [a])
add = ir.create_node("add", [r0, r1], [a])
w = ir.create_node("write", [add], [a])
ir.set_inputs([r0, r1])
ir.set_outputs([w])
```
This defines a pointwise addition, which we can test with numpy:

```python
import numpy as np

# create some data for testing
size = 1024 * 1024
A = lt.Tensor(size)
B = lt.Tensor(size)
C = lt.Tensor(size)
Ap = np.random.randn(size)
Bp = np.random.randn(size)
A.set(Ap)
B.set(Bp)

# compile and run the code
loop_tree = lt.LoopTree(ir)
c = lt.CompiledCuda(loop_tree)
c([A, B, C])

# check the results
C_test = C.to_numpy()
C_ref = Ap + Bp
max_diff = np.max(np.abs(C_test - C_ref))
mean_val = np.mean(np.abs(C_ref))
assert max_diff < 1e-3 * mean_val
```
If we want to optimize this code, we'll need to set a schedule:

```python
# try some other values :) (even weird ones)
inner_size = 512 * 64
vec_size = 4
for n in ir.nodes:
  outer = size // (inner_size * vec_size)
  outer_rem = size % (inner_size * vec_size)

  ir.set_order(n, [
    (v, (outer, outer_rem)), # remainders are well handled
    (v, (inner_size, 0)),
    (v, (vec_size, 0))
    ])
  ir.disable_reuse(n, 2) # loop_tool tries to emit as few loops as possible, but we can prevent that
```

and then parallelize it:

```python
loop_tree = lt.LoopTree(ir)
parallel = set(loop_tree.children(loop_tree.roots[0])) # we'll parallelize the first inner loop
print(loop_tree.dump(lambda x: "// threaded" if x in parallel else "")
```

which will print

```
for a in 8 : L0
 for a' in 32768 : L1 // Threaded
  for a'' in 4 : L2
   %0[a] <- read()
  for a'' in 4 : L4
   %1[a] <- read()
  for a'' in 4 : L6
   %2[a] <- add(%0, %1)
  for a'' in 4 : L8
   %3[a] <- write(%2)
```

And finally, benchmark:

```python
c = lt.CompiledCuda(loop_tree, parallel)
iters = 10000
# warmup
for i in range(50):
  c([A, B, C])
t = time.time()
for i in range(iters - 1):
  c([A, B, C], False)
c([A, B, C])
t_ = time.time()
bytes_moved = (2 + 1) * 4 * size * iters / (t_ - t) / 1e9
pct = bytes_moved / c.bandwidth # loop_tool gives us some useful info
usec = (t_ - t) / iters * 1e6
print(f"{bytes_moved:.2f} GB/sec", f"({100 * pct:.2f}% of peak, {usec:.2f} usec per iter)")
```

You should see something like this:

```
229.85 GB/sec (31.40% of peak, 54.74 usec per iter)
```

If you're curious what the generated code looks like:

```python
print(c.code)
```

```
extern "C" __global__
void kernel(float4* __restrict__ ext_0, float4* __restrict__ ext_1, float4* __restrict__ ext_2) {
 int _tid = blockIdx.x * blockDim.x + threadIdx.x;
 for (int a_0 = 0; a_0 < 8; ++a_0) {
  {
  int a_1 = (_tid / 1) % 32768;
   float4 mem_0[1];
   float4 mem_1[1];
   float4 mem_2[1];
   // unrolling a_2
    ((float*)mem_0)[0 + 0] = ((float*)ext_0)[0 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_0)[1 + 0] = ((float*)ext_0)[1 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_0)[2 + 0] = ((float*)ext_0)[2 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_0)[3 + 0] = ((float*)ext_0)[3 + a_1 * 4 + a_0 * 131072 + 0];
   // unrolling a_2
    ((float*)mem_1)[0 + 0] = ((float*)ext_1)[0 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_1)[1 + 0] = ((float*)ext_1)[1 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_1)[2 + 0] = ((float*)ext_1)[2 + a_1 * 4 + a_0 * 131072 + 0];
    ((float*)mem_1)[3 + 0] = ((float*)ext_1)[3 + a_1 * 4 + a_0 * 131072 + 0];
   // unrolling a_2
    ((float*)mem_2)[0 + 0] = ((float*)mem_0)[0 + 0] + ((float*)mem_1)[0 + 0];
    ((float*)mem_2)[1 + 0] = ((float*)mem_0)[1 + 0] + ((float*)mem_1)[1 + 0];
    ((float*)mem_2)[2 + 0] = ((float*)mem_0)[2 + 0] + ((float*)mem_1)[2 + 0];
    ((float*)mem_2)[3 + 0] = ((float*)mem_0)[3 + 0] + ((float*)mem_1)[3 + 0];
   // unrolling a_2
    ((float*)ext_2)[0 + a_1 * 4 + a_0 * 131072 + 0] = ((float*)mem_2)[0 + 0];
    ((float*)ext_2)[1 + a_1 * 4 + a_0 * 131072 + 0] = ((float*)mem_2)[1 + 0];
    ((float*)ext_2)[2 + a_1 * 4 + a_0 * 131072 + 0] = ((float*)mem_2)[2 + 0];
    ((float*)ext_2)[3 + a_1 * 4 + a_0 * 131072 + 0] = ((float*)mem_2)[3 + 0];
  }
 }
}
```


## Build

To build from source, clone this repo and use `cmake`:

```
git clone https://github.com/facebookresearch/loop_tool.git
mkdir -p build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Python

To build the Python bindings, install `pybind11`:

```
pip install pybind11 # or conda
python setup.py install
```

## Run

If you have CUDA, check out the demo `bench.py` file:

```
python test/bench.py
```

This will sweep a couple of configurations for a simple pointwise addition.
All driven from Python (~100k runs per benchmark), this finds
a 478.58 GB/sec solution on a GP100 device (~65% of peak).


## Extra builds/tests

#### JavaScript

To build a JavaScript target,
specify your `emcc` directory to `cmake`
and rebuild.
This will create two extra files (`loop_tool.js` and `loop_tool.wasm`).

```
EMCC_DIR=$(dirname emcc) cmake ..
make -j$(nproc)
```

## Tests

After building, either run the test binaries or the language tests:

```
./build/test
./build/cuda_test
PYTHONPATH=build python test/test.py
NODE_PATH=build node test/test.js
```


## TODO List

- [x] Interpretted CPU backend
- [x] Scaffold optimized pre-compiled CPU backend
- [ ] ASM CPU backend
- [x] CUDA backend
- [ ] CPP backend
- [ ] Comprehensive tests
- [ ] Tuning script (C++)

## License

loop_tool is MIT licensed, as found in the LICENSE file.
