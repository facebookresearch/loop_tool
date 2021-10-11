/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
let lt = require('loop_tool.js');

lt.onRuntimeInitialized = _ => {
  let N = new lt.Symbol("N");
  let a = new lt.Tensor([128]).to([N]);
  let b = new lt.Tensor([128]).to([N]);
  let c = a.add(b);
  console.log(c);
};

