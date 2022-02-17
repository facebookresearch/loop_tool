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

let n = new lt.Symbol("N");
(async () => {
  let n = new lt.Symbol("N");
  let a = new lt.Tensor(2).to(n);
  a.buffer[0] = 3;
  a.buffer[1] = 2;
  let b = new lt.Tensor(2).to(n);
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
  let [m, n, k] = lt.symbols("M N K");
  let a = new lt.Tensor(100, 200).to(m, k);
  let b = new lt.Tensor(200, 300).to(k, n);
  rand(a.buffer);
  rand(b.buffer);
  let c_ref = mm(a.buffer, b.buffer, 100, 300, 200);
  let c = a.mul(b).sum(k);
  let d = await c.data;
  console.log(c.graphviz);
  console.log(c.shape, c.symbolic_shape);
  //console.log(c_ref, d);
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
    let a = new lt.Tensor(128 * 128).to(n);
    let b = new lt.Tensor(128 * 128).to(n);
    let c = a.add(b);
    c = c.add(b);
    c = c.add(b);
    c = c.add(b);
    c = c.add(b);
    c = c.add(b);
    let d = await c.data;
  }
  console.log(await benchmark(fn), "iters per second");
})();