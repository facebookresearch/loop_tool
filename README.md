# `loop_tool`

`loop_tool` is an experimental, lightweight, and highly-portable linear algebra toolkit.
   
## Install

```
pip install loop_tool_py
python -c 'import loop_tool_py as lt; print(lt.backends())'
```

## Tutorial

A Python notebook tutorial can be found here:
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
./build/loop_tool_test
PYTHONPATH=build python test/test.py
NODE_PATH=build node test/test.js
```

## License

loop_tool is MIT licensed, as found in the LICENSE file.
