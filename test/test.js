/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
let lt = require('loop_tool.js');

function Tensor(...args) {
  let backed_array = new Float32Array([...args].reduce((a,b)=>a*b));
  return new Proxy(new lt._Tensor([...args]), {
    get: function(target, prop, receiver) {
      return backed_array[prop];
    },
    set: function(target, prop, value) {
      backed_array[prop] = value;
    }
  });
}

lt.onRuntimeInitialized = _ => {
  let N = new lt.Symbol("N");
  let a = new lt.Tensor([128]).as([N]);
  let b = new lt.Tensor([128]).as([N]);
  let c = a.add(b)
  try {
    console.log(c.graphviz());
    console.log(c.code());
  } catch (exception) {
    console.error(lt.getExceptionMessage(exception));
  }
  //console.log(c.code());
  return;
  //let k = new Float32Array(128);
  let jjj = Tensor(3, 3);
  jjj[8] = 7;
  console.log(jjj[8]);
  return;
  //let k = Tensor(new Float32Array(128));
  //let c = a.add(b);



  const handler2 = {
    get: function(target, prop, receiver) {
       console.log(prop)
       if (Number.isInteger(+prop)) {
         return prop * 3;
       }
       return "world";
    }
  };
  const target = {}
  const proxy2 = new Proxy(target, handler2);
  console.log(proxy2[9])
  console.log(proxy2["9:5,3"])
  console.log(proxy2[N])
  console.log(N);
};

