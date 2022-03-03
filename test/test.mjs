/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
import * as lt from '../javascript/loop_tool.mjs';
import * as fs from 'fs';

import {
  PerformanceObserver,
  performance
} from 'perf_hooks';

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

try{
(async () => {
  let [m, n, k] = lt.symbols("M N K");
  let a = new lt.Tensor(100, 200).to(m, k);
  let b = new lt.Tensor(200, 300).to(k, n);
  rand(a.buffer);
  rand(b.buffer);
  let c = a.mul(b).sum(k);
  let loop_tree = c.loop_tree;
  console.log(loop_tree.walk().length);
  for (let ref of loop_tree.walk()) {
    const d = loop_tree.depth(ref);
    if (loop_tree.is_loop(ref)) {
      const loop = loop_tree.loop(ref);
      const v = loop.v();
      console.log(" ".repeat(d), "iter", loop_tree.var_name(v));
    } else {
      const node = loop_tree.node(ref);
      console.log(" ".repeat(d), 'node');
    }
  }
})()
} catch(e) {
  console.log(e);
}

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
  const loop_tree = c.loop_tree;
  for (let ref of loop_tree.walk()) {
    if (loop_tree.is_loop(ref)) {
      console.log(loop_tree.depth(ref));
    }
  }
  let d = await c.data;
  console.log("data", d);
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
  console.log(c.hash + '.wasm');
  fs.writeFile(c.hash + '.wasm', c.wasm, _=>{});
  //console.log(c.graphviz);
  let d = await c.data;
  console.log(d);
});

(async () => {
  let n = new lt.Symbol("N");
  const N = 10;
  let a = new lt.Tensor(N).to(n);
  let b = new lt.Tensor(N).to(n);
  rand(a.buffer);
  rand(b.buffer);
  let c = a.add(b);
  c = c.add(b);
  const loop_tree = c.loop_tree;
  let roots = loop_tree.children(-1);
  loop_tree.annotate(roots[0], "unroll");
  console.log(loop_tree.dump());
  c.set_loop_tree(loop_tree);
  console.log(c.hash + '.wasm');
  fs.writeFile(c.hash + '.wasm', c.wasm, _=>{});
  let d = await c.data;
  for (let i = 0; i < N; ++i) {
    if (Math.abs(d[i] - (a.buffer[i] + 2 * b.buffer[i])) > 0.001) {
      console.log("EROR", d[i]);
    }
  }
  console.log(d);
})();

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

async function benchmark(fn, warmup = 100, iters = 10000) {
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
  console.log(await benchmark(fn_wrapped, 10, 100), "iters per second (wrapped)");

  {
    let [m, n, k] = lt.symbols("M N K");
    let a = new lt.Tensor(100, 200).to(m, k);
    let b = new lt.Tensor(200, 300).to(k, n);
    let c = a.mul(b).sum(k);
    let [mem_map, fn] = await c.compile();
    let iter_sec = await benchmark(fn, 10, 100);
    console.log(iter_sec, "mm iters per second (pure fn)", `${100 * 200 * 300 * 2 * iter_sec / 1e9} gflops`);
  }

})();

(async () => {
  let m = lt.symbol("m");
  let a = lt.rand(128).to(m);
  let b = a.sum(m);
  const d = new Float32Array(1);
  d.set(await b.data);
  let tree = b.loop_tree;
  const roots = tree.children(-1);
  let new_tree = tree.annotate(roots[0], "unroll");
  tree.delete();
  tree = new_tree;
  b.set_loop_tree(tree);
  const e = new Float32Array(1);
  console.log(b.data);
  e.set(await b.data);
  console.log("diff", d, e);
})();

