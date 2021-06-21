# `loop_tool`

`loop_tool` is a lightweight, highly-portable IR for compiling N-dimensional data computation.

## Build

To build from source, clone this repo and use `cmake`:

```
git clone https://github.com/facebookresearch/loop_tool.git
mkdir -p build; cd build
cmake ..
make -j$(nproc)
```

#### Python

To build the Python bindings, install `pybind11`:

```
pip install pybind11 # or conda
python setup.py install
```

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


## TODO

- [x] Interpretted CPU backend
- [x] Scaffold optimized pre-compiled CPU backend
- [ ] ASM CPU backend
- [x] CUDA backend
- [ ] CPP backend
- [-] Comprehensive tests
- [ ] Tuning script (C++)

## License

loop_tool is MIT licensed, as found in the LICENSE file.
