/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
import * as lt from '../javascript/loop_tool.mjs';

import {
  PerformanceObserver,
  performance
} from 'perf_hooks';

let e = (new lt.Expr(3)).add(new lt.Symbol("k").expr());
console.log(e);
console.log(e.dump());

(async () => {
  let n = new lt.Symbol("N");
  let k = new lt.Symbol("K");
  let no = new lt.Symbol("No");
  let a = new lt.Tensor(10).to(n);
  rand(a.buffer);
  let b = new lt.Tensor(3).to(k);
  rand(b.buffer);
  a = a.to(no, k, [[n.expr(), no.expr().add(k.expr())]]);
  let c = a.mul(b).sum(k);
  console.log(c.shape);
  let d = await c.data;
  //console.log("data", d);
})();

(async () => {
  let n = new lt.Symbol("N");
  let a = new lt.Tensor(2).to(n);
  a.buffer[0] = 3;
  a.buffer[1] = 2;
  let b = new lt.Tensor(2).to(n);
  b.set(new Float32Array([4, 9]));
  let c = a.add(b);
  c = c.add(b);
  //console.log(c.graphviz);
  let d = await c.data;
  console.log(d);
})();

function cmp(a, b) {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; ++i) {
    if (Math.abs(a[i] - b[i]) > 0.001) {
      console.log(a[i], b[i], "at index", i);
      return false;
    }
  }
  return true;

}

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
  let [m, n, k] = lt.symbols("M N K");
  let a = new lt.Tensor(100, 200).to(m, k);
  let b = new lt.Tensor(200, 300).to(k, n);
  rand(a.buffer);
  rand(b.buffer);
  let c_ref = mm(a.buffer, b.buffer, 100, 300, 200);
  let c = a.mul(b).sum(k);
  let d = await c.data;
  //console.log(c.graphviz);
  console.log(c.shape, c.symbolic_shape);
  if (cmp(c_ref, d)) {
    console.log("results look good");
  } else {
    console.log("ERROR!");
  }
})();

async function benchmark(fn, warmup = 10, iters = 10) {
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
  const fn_wrapped = async () => {
    let n = new lt.Symbol("N");
    let a = new lt.Tensor(128 * 128).to(n);
    let b = new lt.Tensor(128 * 128).to(n);
    let c = a.add(b);
    let d = await c.data;
  }
  let n = new lt.Symbol("N");
  let a = new lt.Tensor(128 * 128).to(n);
  let b = new lt.Tensor(128 * 128).to(n);
  let c = a.add(b);
  let [mem_map, fn] = await c.compile();
  const fn_mem = async () => {
    for (let k of Object.keys(mem_map)) {
      if (k == c._id) {
        continue;
      }
      mem_map[k].fill(1);
    }
    fn();
  }
  console.log(await benchmark(fn), "iters per second (pure fn)");
  console.log(await benchmark(fn_mem), "iters per second (fn + fill inputs)");
  console.log(await benchmark(fn_wrapped), "iters per second (wrapped)");
});
