/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
let lt = require('loop_tool.js');
const {
    PerformanceObserver,
    performance
} = require('perf_hooks');


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

function symbols(str) {
    let out = [];
    for (let k of str.split(' ')) {
        out.push(new lt.Symbol(str));
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
            for (let inp of this._inputs) {
                let inp_d = new Float32Array(mem.buffer, offset, inp.numel);
                inp_d.set(inp.buffer);
                offset += inp.numel * 4;
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

lt.onRuntimeInitialized = _ => {
    (async () => {
        let n = new lt.Symbol("N");
        let a = new Tensor(2).to(n);
        a.buffer[0] = 3;
        a.buffer[1] = 2;
        let b = new Tensor(2).to(n);
        b.set(new Float32Array([4, 9]));
        let c = a.add(b);
        c = c.add(b);
        console.log(c.graphviz);
        let d = await c.data;
        console.log(d);
    })();

    function rand(array) {
        for (let i = 0; i < array.length; ++i) {
            array[i] = Math.random();
        }
    }

    function mm(a, b, m, n, k) {
        const c = new Float32Array(m * n);
        for (let m_ = 0; m_ < m; ++m_) {
            for (let n_ = 0; n_ < n; ++n_) {
                for (let k_ = 0; k_ < k; ++k_) {
                    c[m_ * n + n_] += a[m_ * k + k_] * b[k_ * n + n_];
                }
            }
        }
        return c;
    }

    (async () => {
        let [m, n, k] = symbols("M N K");
        let a = new Tensor(100, 200).to(m, k);
        let b = new Tensor(200, 300).to(k, n);
        rand(a.buffer);
        rand(b.buffer);
        let c_ref = mm(a.buffer, b.buffer, 100, 300, 200);
        let c = a.mul(b).sum(k);
        let d = await c.data;
        console.log(c.shape, c.symbolic_shape);
        console.log(c_ref, d);
    })();

    async function benchmark(fn, warmup = 10, iters = 100) {
        for (let i = 0; i < warmup; ++i) {
            await fn();
        }
        let t0 = performance.now();
        for (let i = 0; i < iters; ++i) {
            await fn();
        }
        let t1 = performance.now();
        return 1e3 * iters / (t1 - t0);
    }

    (async () => {
        const fn = async () => {
            let n = new lt.Symbol("N");
            let a = new Tensor(128).to(n);
            let b = new Tensor(128).to(n);
            let c = a.add(b);
            c = c.add(b);
            c = c.add(b);
            c = c.add(b);
            c = c.add(b);
            let d = await c.data;
        }
        console.log(await benchmark(fn), "iters per second");
    })();
};