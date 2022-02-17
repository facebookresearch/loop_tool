/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
import Module from '../build/libloop_tool.js';

const lt = await Module();

class CompilationCache {
  constructor(max = 2000) {
    this.max = max;
    this.hash_list = [];
    this.cache = {};
  };

  async compile(tensor) {
    const h = tensor.hash();
    if (h in this.cache) {
      return this.cache[h];
    }
    if (this.hash_list.length == this.max) {
      const remove = this.hash_list[0];
      this.hash_list = this.hash_list.slice(1);
      delete cache[remove];
    }
    this.hash_list.push(h);
    const m = await WebAssembly.compile(tensor.wasm()).catch(e => {
      throw e;
    });
    const instance = await WebAssembly.instantiate(m, {}).catch(e => {
      throw e;
    });
    this.cache[h] = [instance.exports.mem, instance.exports.fn];
    return this.cache[h];
  }
};

let cc = new CompilationCache();

let Symbol = lt.Symbol;

function symbols(str) {
  let out = [];
  for (let k of str.split(' ')) {
    out.push(new lt.Symbol(k));
  }
  return out;
}

class Tensor {
  // either a size, an array or an lt.Tensor
  constructor(...args) {
    this._inputs = [];
    this._data = null;
    if (args.length == 1 && args[0].constructor == lt.Tensor) {
      this._tensor = args[0];
      this._compute = true;
    }
    if (this._tensor === undefined) {
      this._tensor = new lt.Tensor([...args]);
      this._compute = false;
    }
  }

  get buffer() {
    if (this._compute) {
      throw "Cannot access buffer of compute tensor";
    }
    if (!this._data) {
      this._data = new Float32Array(this._tensor.numel());
    }
    return this._data;
  }

  set(new_data) {
    if (this._compute) {
      throw "Cannot set buffer of compute tensor";
    }
    if (new_data.length != this._tensor.numel()) {
      throw `Cannot set buffer to size ${new_data.length}, expected ${this._tensor.numel()}`;
    }
    this._data = new_data;
  }

  get data() {
    return (async () => {
      if (this._data) {
        return this._data;
      }

      let [mem, fn] = await cc.compile(this._tensor);
      let offset = 0;
      let seen = {};
      for (let inp of this._inputs) {
        let inp_off = offset;
        if (inp in seen) {
          inp_off = seen[inp];
        } else {
          offset += inp.numel * 4;
        }
        let inp_d = new Float32Array(mem.buffer, inp_off, inp.numel);
        inp_d.set(inp.buffer);
        seen[inp] = offset;
      }
      fn();
      this.data_ = new Float32Array(mem.buffer, offset, this.numel);
      return this.data_;
    })();
  }

  get shape() {
    return this._tensor.shape();
  }

  get symbolic_shape() {
    return this._tensor.symbolic_shape();
  }

  get graphviz() {
    return this._tensor.graphviz();
  }

  get code() {
    return this._tensor.code();
  }

  collect_inputs(...ts) {
    let inputs = [];
    for (let t of ts) {
      if (t._inputs.length) {
        inputs = inputs.concat(t._inputs);
      } else {
        inputs.push(t);
      }
    }
    return inputs;
  }

  sum(...syms) {
    let out_t = new Tensor(this._tensor.sum(syms));
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  mul(t) {
    let out_t = new Tensor(this._tensor.mul(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  add(t) {
    let out_t = new Tensor(this._tensor.add(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  to(...syms) {
    this._tensor = this._tensor.as(syms);
    return this;
  }

  get numel() {
    return this._tensor.numel();
  }
};

export {
  Tensor,
  Symbol,
  symbols
};