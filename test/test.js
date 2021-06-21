/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
let loop_tool = require('loop_tool.js');
loop_tool.onRuntimeInitialized = _ => {
  let ir = new loop_tool.IR();
  let a = ir.create_var("a");
  let r0 = ir.create_node("read", [], [a]);
  let r1 = ir.create_node("read", [], [a]);
  let add = ir.create_node("add", [r0, r1], [a]);
  let write = ir.create_node("write", [add], [a]);
  ir.set_inputs([r0, r1]);
  ir.set_outputs([write]);
  let lt = new loop_tool.LoopTree(ir);
  console.log(lt.dump());
};

