/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
import Module from '../build/libloop_tool.mjs';

const lt = await Module();

const Symbol = lt.Symbol;
const Expr = lt.Expr;
const size = lt.size;
const expr = (...args) => { return new lt.Expr(...args); }
const tensor = (...args) => { return new Tensor(...args); }
const symbol = (...args) => { return new lt.Symbol(...args); }

const getExceptionMessage = lt.getExceptionMessage

function symbols(str) {
  let out = [];
  for (let k of str.split(' ')) {
    out.push(new lt.Symbol(k));
  }
  return out;
}

class CompilationCache {
  constructor(max = 2000) {
    this.max = max;
    this.hash_list = [];
    this.loop_tree_cache = {};
    this.instance_cache = {};
  };

  async compile(tensor) {
    const h = tensor.hash();
    if (h in this.instance_cache) {
      return this.instance_cache[h];
    }
    if (this.hash_list.length == this.max) {
      const remove = this.hash_list[0];
      this.evict(remove);
    }
    this.hash_list.push(h);

    if (!(h in this.loop_tree_cache)) {
      this.loop_tree_cache[h] = tensor.loop_tree();
    }
    const loop_tree = this.loop_tree_cache[h];
    const m = await WebAssembly.compile(loop_tree.wasm()).catch(e => {
      throw e;
    });
    const instance = await WebAssembly.instantiate(m, {}).catch(e => {
      throw e;
    });
    this.instance_cache[h] = [instance.exports.mem, instance.exports.fn];
    return this.instance_cache[h];
  }

  evict(h) {
    const idx = this.hash_list.indexOf(h);
    if (idx < 0) {
      return;
    }
    this.hash_list = this.hash_list.splice(idx, 1);
    delete this.instance_cache[h];
    delete this.loop_tree_cache[h];
  }

  loop_tree(tensor) {
    const h = tensor.hash();
    if (!(h in this.loop_tree_cache)) {
      return tensor.loop_tree();
    }
    return this.loop_tree_cache[h];
  }

  set_loop_tree(tensor, loop_tree) {
    const h = tensor.hash();
    this.evict(h);
    this.loop_tree_cache[h] = loop_tree;
  }

};

let _tensor_id = 0;
let cc = new CompilationCache();

function fill(val) {
  const t = new Tensor();
  t.buffer[0] = val;
  return t;
}

function rand(...args) {
  const t = new Tensor(...args);
  for (let i = 0; i < t.buffer.length; ++i) {
    t.buffer[i] = Math.random();
  }
  return t;
}

function convolve(x, w, spatial_dims, window_dims, stride=1) {
  if (spatial_dims.length != window_dims.length) {
    throw 'invalid call to conv: mismatched dim length';
  }
  const new_spatial = [];
  const constraints = [];
  for (let i = 0; i < spatial_dims.length; ++i) {
    const dim = spatial_dims[i];
    const k = window_dims[i];
    const new_dim = new Symbol(dim.name() + "o");
    new_spatial.push(new_dim);
    constraints.push([dim.expr(), new_dim.expr().mul(expr(stride)).add(k.expr())]);
  }

  const outer_dims = [];
  const reduction_dims = [...window_dims];
  const x_ids = x.symbolic_shape.map(d=>d.id());
  const spatial_ids = spatial_dims.map(d=>d.id());
  const window_ids = window_dims.map(d=>d.id());

  for (let dim of x.symbolic_shape) {
    if (spatial_ids.includes(dim.id())) {
      continue;
    }
    outer_dims.push(dim);
  }

  for (let dim of w.symbolic_shape) {
    if (window_ids.includes(dim.id())) {
      if (x_ids.includes(dim.id())) {
        throw `invalid window dim: ${dim.name()}`
      }
      continue;
    }
    if (x_ids.includes(dim.id())) {
      reduction_dims.push(dim);
      continue;
    }
    outer_dims.push(dim);
  }

  const im2col = x.to(...outer_dims, ...new_spatial, ...window_dims, constraints);
  return im2col.mul(w).sum(...reduction_dims);
}

function relu(x) {
  return x.max(0);
}

function maxpool(x, spatial_dims, k) {
  const constraints = [];
  const outer_dims = [];
  const new_dims = [];
  const window_dims = [];
  for (let dim of spatial_dims) {
    const new_dim = symbol(dim.name() + "p");
    const new_k = symbol(dim.name() + "k");
    new_dims.push(new_dim);
    window_dims.push(new_k);
    const c = new_dim.expr().mul(expr(k)).add(new_k.expr());
    constraints.push([dim.expr(), c]);
    constraints.push([size(new_k), expr(k)])
  }
  const spatial_ids = spatial_dims.map(d=>d.id());
  for (let dim of x.symbolic_shape) {
    if (spatial_ids.includes(dim.id())) {
      continue;
    }
    outer_dims.push(dim);
  }
  return x.to(...outer_dims, ...new_dims, ...window_dims, constraints).max(...window_dims);
}

function avgpool(x, spatial_dims, k) {
  const constraints = [];
  const outer_dims = [];
  const new_dims = [];
  const window_dims = [];
  console.log("K IS", k, x.symbolic_shape);
  let d = fill(1 / (k*k));
  x = x.mul(d);
  for (let dim of spatial_dims) {
    const new_dim = symbol(dim.name() + "a");
    const new_k = symbol(dim.name() + "k");
    new_dims.push(new_dim);
    window_dims.push(new_k);
    const c = new_dim.expr().mul(expr(k)).add(new_k.expr());
    constraints.push([dim.expr(), c]);
    constraints.push([size(new_k), expr(k)])
  }
  const spatial_ids = spatial_dims.map(d=>d.id());
  for (let dim of x.symbolic_shape) {
    if (spatial_ids.includes(dim.id())) {
      continue;
    }
    outer_dims.push(dim);
  }
  return [x.to(...outer_dims, ...new_dims, ...window_dims, constraints).sum(...window_dims), d];
}


class Tensor {
  // either a size, an array or an lt.Tensor
  constructor(...args) {
    this._id = _tensor_id++;
    this._inputs = [];
    this._data = null;
    if (args.length == 1 && args[0].constructor == lt.Tensor) {
      this._tensor = args[0];
      this._compute = true;
    }
    if (this._tensor === undefined) {
      this._tensor = new lt.Tensor(args);
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

  async compile() {
    try {
      let [mem, fn] = await cc.compile(this._tensor);
      let offset = 0;
      let mem_map = {};
      for (let inp of this._inputs) {
        if (inp._id in mem_map) {
          continue;
        }
        let inp_d = new Float32Array(mem.buffer, offset, inp.numel);
        inp_d.set(inp.buffer);
        offset += inp.numel * 4;
        mem_map[inp._id] = inp_d;
      }
      mem_map[this._id] = new Float32Array(mem.buffer, offset, this.numel);
      return [mem_map, fn];
    } catch(e) {
			throw e;
    }
  }

  get wasm() {
    return this.loop_tree.wasm();
  }

  get flops() {
    return this.loop_tree.flops();
  }

  benchmark_impl(ms, fn) {
		let iters = 1;
    let t = performance.now();
    let d = 0;
    do {
      for (let i = 0; i < iters; ++i) {
				fn();
			}
			d = performance.now() - t;
      iters *= 2;
    } while (d < ms);
		return 1e3 * (iters - 1) / d;
	}

  async benchmark(ms=50, warmup_ms=10) {
    try {
      const [m, fn] = await this.compile();
      this.benchmark_impl(warmup_ms, fn);
      return this.flops * this.benchmark_impl(ms, fn);
    } catch(e) {
      throw lt.getExceptionMessage(e);
    }
	}

  get hash() {
    return this._tensor.hash();
  }

  get id() {
    return this._id;
  }

  get data() {
    return (async () => {
      if (this._data !== null) {
        return this._data;
      }
      let [mem_map, fn] = await this.compile();
      this.data_ = mem_map[this._id];
      fn();
      return this.data_;
    })().catch(e => {
      throw e;
    });
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
    const regex = /void fn_[0-9]+\(/i;
    const c = this.loop_tree.code();
    return c.replace(regex, 'void fn(');
  }

  get loop_tree() {
    return cc.loop_tree(this._tensor);
  }

  set_loop_tree(loop_tree) {
    cc.set_loop_tree(this._tensor, loop_tree);
    this.data_ = null;
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

  mul(t) {
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.mul(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  div(t) {
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.div(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  add(t) {
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.add(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  sub(t) {
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.sub(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  minus(t) {
    return this.sub(t);
  }

  min_reduce(...syms) {
    let out_t = new Tensor(this._tensor.min_reduce(syms));
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  max_reduce(...syms) {
    let out_t = new Tensor(this._tensor.max_reduce(syms));
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  min(...args) {
    if (args[0].constructor == lt.Symbol) {
      return this.min_reduce(...args);
    }
    let t = args[0];
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.min(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  max(...args) {
    if (args[0].constructor == lt.Symbol) {
      return this.max_reduce(...args);
    }
    let t = args[0];
    t = t.constructor === Number ? fill(t) : t;
    let out_t = new Tensor(this._tensor.max(t._tensor));
    out_t._inputs = this.collect_inputs(this, t);
    return out_t;
  }

  sum(...syms) {
    let out_t = new Tensor(this._tensor.sum(syms));
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  neg() {
    let out_t = new Tensor(this._tensor.neg());
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  abs() {
    let out_t = new Tensor(this._tensor.abs());
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  sqrt() {
    let out_t = new Tensor(this._tensor.sqrt());
    out_t._inputs = this.collect_inputs(this);
    return out_t;
  }

  to(...syms) {
    const l = syms.length;
    if (l && syms[l-1].constructor != lt.Symbol) {
      const constraints = syms[l-1];
      if (constraints.length && constraints[0][0].constructor != lt.Expr) {
        throw 'invalid arguments! requires to(...syms) or to(...syms, constraints[])';
      }
      let out_t = new Tensor(this._tensor.to(syms.slice(0, -1), constraints));
      out_t._inputs = this.collect_inputs(this);
      return out_t;
    }
    this._tensor = this._tensor.as(syms);
    return this;
  }

  transpose(...syms) {
		let out_t = new Tensor(this._tensor.transpose(syms));
		out_t._inputs = this.collect_inputs(this);
		return out_t;
	}

  get numel() {
    return this._tensor.numel();
  }
};

export {
  // raw constructors
  Tensor,
  Symbol,
  Expr,
  // function based constructors
  tensor,
  fill,
  rand,
	convolve,
  maxpool,
  avgpool,
  relu,
  symbol,
  symbols,
  expr,
  size,
  getExceptionMessage
};
