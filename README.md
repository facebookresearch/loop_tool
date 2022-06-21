# `loop_tool`

A tiny linear algebra code-generator and optimization toolkit.  Try it out here: http://loop-tool.glitch.me


![](https://user-images.githubusercontent.com/4842908/174701426-5e479bad-821e-4395-ae1a-91c48776efd9.gif)



## Installation

### C++:
```bash
git clone https://github.com/facebookresearch/loop_tool.git
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Python:
```bash
pip install loop_tool
```

### JavaScript:
```bash
curl -O -L https://github.com/facebookresearch/loop_tool/raw/main/javascript/lt.mjs.gz
gunzip lt.mjs.gz
```

## Import
### C++:
```cpp
#include <loop_tool/loop_tool.h>
namespace lt = loop_tool;
```
### Python:
```python
import loop_tool as lt
```
### JavaScript:
```javascript
import * as lt from './lt.mjs';
```

## Usage

### C++:
```cpp
#include <loop_tool/loop_tool.h>
namespace lz = ::loop_tool::lazy;

auto mm = [](lz::Tensor A, lz::Tensor B) {
  lz::Symbol M, N, K;
  auto C = A.as(M, K) * B.as(K, N);
  return C.sum(K);
};

lz::Tensor A(128, 128);
lz::Tensor B(128, 128);
rand(A.data<float>(), 128 * 128);
rand(B.data<float>(), 128 * 128);

auto C = mm(A, B);
std::cout << C.data<float>()[0];
```

### Python:
```python
import loop_tool as lt
import numpy as np

def mm(a, b):
    m, n, k = lt.symbols("m n k")
    return (a.to(m, k) * b.to(k, n)).sum(k)

A_np = np.random.randn(128, 128)
B_np = np.random.randn(128, 128)
A = lt.Tensor(A_np)
B = lt.Tensor(B_np)

C = mm(A, B)
print(C.numpy()[0])
```
### JavaScript:
```javascript
import * as lt from './lt.mjs';

function mm(A, B) {
  const [m, n, k] = lt.symbols("m n k");
  return A.to(m, k).mul(B.to(k, n)).sum(k);
}

const A = lt.tensor(128, 128);
const B = lt.tensor(128, 128);
A.set(new Float32Array(128 * 128, 1));
B.set(new Float32Array(128 * 128, 1));

const C = mm(A, B);
console.log(await C.data()[0]);
```

## Overview

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
    - Lowered to various backends (currently C, WASM)

Additionally, a curses-based UI is provided (in Python `lt.ui(tensor)`) to interactively optimize loop structures on the fly:

https://user-images.githubusercontent.com/4842908/172877041-fd4b9ed8-7164-49f6-810f-b14f169e2ca9.mp4

## Arithmetic Support

`loop_tool` is built on fixed-range loop iterations over quasi-linear index equations.  Within these loops, the following operations are currently supported:

- `copy`
- `add`
- `subtract`
- `multiply`
- `divide`
- `min`
- `max`
- `log`
- `exp`
- `sqrt`
- `abs`
- `negate`
- `reciprocal`

It is straight-forward to add arithmetic support if needed.

With the constraint-based quasi-linear indexing system, far more complex operations can easily be defined, such as

- Padding
- Concatenation
- Matrix multiplication
- Convolution (group, strided, dilated)
- Pooling (max, average)

## License

loop_tool is MIT licensed, as found in the LICENSE file.
