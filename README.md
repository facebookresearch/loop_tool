# `loop_tool`

`loop_tool` is an experimental loop-based computation toolkit.
Building on the fact that many useful operations (in linear algebra, neural networks, and media processing)
can be written as highly optimized bounded loops,
`loop_tool` is composed of two ideas:

1. A lazy symbolic frontend
    - Extension of typical eager interfaces (e.g. [Numpy](https://numpy.org) or earlier [PyTorch](https://pytorch.org))
    - Symbolic shape deduction (including input shapes)
    - Transparent JIT compilation
2. A simple functional IR
    - Optimized through local node-level annotations
    - Lowered to various backends (currently CPU and CUDA)

## Install

`loop_tool` clocks in under 400KB for both Linux and Mac OS.

```
pip install loop_tool_py
```

Verify the installation worked and determine which backends are supported:

```
python -c 'import loop_tool_py as lt; print(lt.backends())'
```

## Usage

Generally,
```python
for i in range(I):
  for j in range(J):
    c[j] += a[i, j] * b[i]
```
is equivalent to
```python
I, J = lt.Symbol("I"), lt.Symbol("J")
a = lt.Tensor(I, J)
b = lt.Tensor(I)
c = (a * b).sum(I)
```

There are a couple of ways to use `loop_tool`:

As an eager linear algebra API

```python
import loop_tool_py as lt
import numpy as np

# tensor of size 128, initialized with numpy
X = lt.Tensor(128).set(np.random.randn(128))

# name the dimension and then reduce over it
N = lt.Symbol("N")
Y = X.to(N).sum(N)

assert np.allclose(Y.numpy(), np.sum(X.numpy()))
```

As a lazy linear algebra API

```python
# tensor of size K, uninitialized
K = lt.Symbol("K")
Z = lt.Tensor(K)

# use the uninitialized tensor (and rename dimensions)
W = (Z.to(N) * X.to(N)).sum(N)

# derive information about symbolic shapes
W.unify()
assert Z.shape[0] == 128

# tensor Z initialized later
Z.set(np.random.randn(Z.shape[0]))

assert np.allclose(W.numpy(), np.sum(X.numpy() * Z.numpy()))
```

As an optimization toolkit

```python
# dump information about the computation
print(W.loop_tree)
# for N_6 in 128 : L0
#  %0[N_6] <- read()
#  %1[N_6] <- read()
#  %2[N_6] <- multiply(%0, %1)
#  %3[] <- add(%2)
# %4[] <- write(%3)

# schedule different loop orders
ir = W.ir
v = ir.vars[0]
for n in ir.nodes:
  ir.set_order(n, [(v, (8, 0)), (v, (16, 0))])
  # force multiply to have a different inner loop
  if "multiply" in ir.dump(n):
    ir.disable_reuse(n, 1)
W.set(ir)

print(W.loop_tree)
# for N_6 in 8 : L0
#  for N_6' in 16 : L1
#   %0[N_6] <- read()
#   %1[N_6] <- read()
#   %2[N_6] <- multiply(%0, %1)
#  for N_6' in 16 : L5
#   %3[] <- add(%2)
# for N_6 in 8 : L7
#  for N_6' in 16 : L8
#   %4[] <- write(%3)

new_X = lt.Tensor(128).set(np.random.randn(128))
new_Z = lt.Tensor(K)
new_W = (new_Z.to(N) * new_X.to(N)).sum(N)
new_W.unify()
new_Z.set(np.random.randn(128))

# same compute, same loop_tree
assert str(new_W.loop_tree) == str(W.loop_tree)

assert np.allclose(new_W.numpy(), np.sum(new_X.numpy() * new_Z.numpy()))
```

As a CUDA code generator

```python
if "cuda" in lt.backends():
  lt.set_default_backend("cuda")
  lt.set_default_hardware("cuda")
  W = lt.Tensor(128).to(N) + lt.Tensor(128).to(N)
  print(W.compiled.code)

  loop_tree = W.loop_tree
  loop_tree.annotate(loop_tree.loops[0], "parallel")
  W.set(loop_tree)
  print(W.compiled.code)
```
Almost all examples above have nearly identical C++ interfaces, e.g.

```cpp
namespace lz = ::loop_tool::lazy;

lz::Tensor A(128);
lz::Tensor B(128);
A.data<float>()[0] = 1.3; // rather than "set()"
  
auto N = lz::Symbol("N");
auto C = A.as(N) + B.as(N);
  
auto lt = C.loop_tree();
std::cout << lt.dump() << "\n";

lt.annotate(lt.loops()[0], "parallel");
C.set(lt);
```

## Tutorial

For zero-install interactive usage examples,
a step by step Python notebook tutorial can be found here:
https://github.com/facebookresearch/loop_tool/blob/main/tutorial.ipynb

## Build C++ API from source

To build the C++ API from source, clone this repo and use `cmake`:

```
git clone https://github.com/facebookresearch/loop_tool.git
mkdir -p build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

#### Python

To build the Python bindings from source, install `pybind11`:

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
All driven from Python (~100k runs per benchmark), this should be able to find a
schedule that hits ~70% of peak bandwidth regardless of GPU.


## Extra builds/tests

#### JavaScript (work in progress)

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
./build/loop_tool_test
PYTHONPATH=build python test/test.py
NODE_PATH=build node test/test.js
```

## License

loop_tool is MIT licensed, as found in the LICENSE file.
