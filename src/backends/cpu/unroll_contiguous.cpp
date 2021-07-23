/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "loop_tool/backend.h"
#include "loop_tool/error.h"

#include <cstring>
#include <iostream>

using namespace loop_tool;

InnerFnType gen_vec_read(const LoopTree &lt, LoopTree::TreeRef ref,
                         const Allocation &alloc, int N) {
  int external_memory = -1;
  for (auto i = 0; i < lt.ir.inputs().size(); ++i) {
    if (lt.ir.inputs()[i] == lt.node(ref).node) {
      external_memory = i;
    }
  }
  ASSERT(external_memory > -1 && "No input found!");

  auto idx_fn = gen_idx_func(lt, alloc, ref);
  auto alloc_read = alloc;
  // TODO this is a hacky way to ensure all variables are in the indexing
  alloc_read.lca = -1;
  auto read_idx_fn = gen_idx_func(lt, alloc_read, ref);
  auto inp_memory = alloc.idx + lt.ir.inputs().size() + lt.ir.outputs().size();
  auto depth = lt.node(lt.node(ref).parent).depth;

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < depth; ++i) {
      if (tails[i]) {
        return;
      }
    }
    auto out_off = idx_fn(indices);
    auto inp_off = read_idx_fn(indices);
    auto out_mem = (float *)memory[inp_memory];
    auto in_mem = (float *)memory[external_memory];
    auto i = 0;
    memcpy(&out_mem[out_off], &in_mem[inp_off], N * sizeof(float));
  };
}

InnerFnType
gen_vec_write(const LoopTree &lt,
              const std::unordered_map<IR::NodeRef, Allocation> &allocs,
              LoopTree::TreeRef ref, int N) {
  int external_memory = -1;
  auto tree_node = lt.node(ref);
  for (auto i = 0; i < lt.ir.outputs().size(); ++i) {
    if (lt.ir.outputs()[i] == tree_node.node) {
      external_memory = i + lt.ir.inputs().size();
    }
  }
  ASSERT(external_memory > -1 && "No output found!");

  const auto &n = lt.ir.node(tree_node.node);
  ASSERT(n.inputs().size() == 1);
  ASSERT(n.outputs().size() == 0);

  auto inp = n.inputs()[0];

  auto inp_idx_fn = gen_idx_func(lt, allocs.at(inp), ref);
  auto out_idx_fn = gen_idx_func(lt, allocs.at(tree_node.node), ref);
  auto alloc = allocs.at(tree_node.node);
  auto input_memory =
      allocs.at(inp).idx + lt.ir.inputs().size() + lt.ir.outputs().size();
  auto depth = lt.node(lt.node(ref).parent).depth;

  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < depth; ++i) {
      if (tails[i]) {
        return;
      }
    }
    auto out_off = out_idx_fn(indices);
    auto inp_off = inp_idx_fn(indices);
    auto in_mem = (float *)memory[input_memory];
    auto out_mem = (float *)memory[external_memory];
    memcpy(&out_mem[out_off], &in_mem[inp_off], N * sizeof(float));
    // for (auto i = 0; i < N; ++i) {
    //  out_mem[out_off + i] = in_mem[inp_off + i];
    //}
  };
}

InnerFnType
gen_vec_add(const LoopTree &lt,
            const std::unordered_map<IR::NodeRef, Allocation> &allocs,
            LoopTree::TreeRef ref, int N) {
  auto tree_node = lt.node(ref);
  const auto &n = lt.ir.node(tree_node.node);
  auto depth = lt.node(lt.node(ref).parent).depth;

  std::vector<std::pair<std::function<size_t(int[MAX_DEPTH])>, int>> inputs;
  std::pair<std::function<size_t(int[MAX_DEPTH])>, int> output;

  auto mem_off = lt.ir.inputs().size() + lt.ir.outputs().size();
  for (auto &inp_ref : n.inputs()) {
    const auto &alloc = allocs.at(inp_ref);
    inputs.emplace_back(gen_idx_func(lt, alloc, ref), alloc.idx + mem_off);
  }
  auto out_alloc = allocs.at(tree_node.node);

  output =
      std::make_pair(gen_idx_func(lt, out_alloc, ref), out_alloc.idx + mem_off);
  return [=](const std::vector<void *> &memory, int indices[MAX_DEPTH],
             int tails[MAX_DEPTH]) {
    for (auto i = 0; i < depth; ++i) {
      if (tails[i]) {
        return;
      }
    }

    auto out_off = output.first(indices);
    auto in0_off = inputs.at(0).first(indices);
    auto in1_off = inputs.at(1).first(indices);
    float *out_mem = ((float *)memory[output.second]);
    float *in_mem0 = ((float *)memory[inputs.at(0).second]);
    float *in_mem1 = ((float *)memory[inputs.at(1).second]);
    auto i = 0;
    for (; (i + 15) < N; i += 16) {
      auto v_0 = in_mem0[in0_off + i + 0] + in_mem1[in1_off + i + 0];
      auto v_1 = in_mem0[in0_off + i + 1] + in_mem1[in1_off + i + 1];
      auto v_2 = in_mem0[in0_off + i + 2] + in_mem1[in1_off + i + 2];
      auto v_3 = in_mem0[in0_off + i + 3] + in_mem1[in1_off + i + 3];
      auto v_4 = in_mem0[in0_off + i + 4] + in_mem1[in1_off + i + 4];
      auto v_5 = in_mem0[in0_off + i + 5] + in_mem1[in1_off + i + 5];
      auto v_6 = in_mem0[in0_off + i + 6] + in_mem1[in1_off + i + 6];
      auto v_7 = in_mem0[in0_off + i + 7] + in_mem1[in1_off + i + 7];
      auto v_8 = in_mem0[in0_off + i + 8] + in_mem1[in1_off + i + 8];
      auto v_9 = in_mem0[in0_off + i + 9] + in_mem1[in1_off + i + 9];
      auto v_10 = in_mem0[in0_off + i + 10] + in_mem1[in1_off + i + 10];
      auto v_11 = in_mem0[in0_off + i + 11] + in_mem1[in1_off + i + 11];
      auto v_12 = in_mem0[in0_off + i + 12] + in_mem1[in1_off + i + 12];
      auto v_13 = in_mem0[in0_off + i + 13] + in_mem1[in1_off + i + 13];
      auto v_14 = in_mem0[in0_off + i + 14] + in_mem1[in1_off + i + 14];
      auto v_15 = in_mem0[in0_off + i + 15] + in_mem1[in1_off + i + 15];
      out_mem[out_off + i + 0] += v_0;
      out_mem[out_off + i + 1] += v_1;
      out_mem[out_off + i + 2] += v_2;
      out_mem[out_off + i + 3] += v_3;
      out_mem[out_off + i + 4] += v_4;
      out_mem[out_off + i + 5] += v_5;
      out_mem[out_off + i + 6] += v_6;
      out_mem[out_off + i + 7] += v_7;
      out_mem[out_off + i + 8] += v_8;
      out_mem[out_off + i + 9] += v_9;
      out_mem[out_off + i + 10] += v_10;
      out_mem[out_off + i + 11] += v_11;
      out_mem[out_off + i + 12] += v_12;
      out_mem[out_off + i + 13] += v_13;
      out_mem[out_off + i + 14] += v_14;
      out_mem[out_off + i + 15] += v_15;
    }
    for (; (i + 7) < N; i += 8) {
      auto v_0 = in_mem0[in0_off + i + 0] + in_mem1[in1_off + i + 0];
      auto v_1 = in_mem0[in0_off + i + 1] + in_mem1[in1_off + i + 1];
      auto v_2 = in_mem0[in0_off + i + 2] + in_mem1[in1_off + i + 2];
      auto v_3 = in_mem0[in0_off + i + 3] + in_mem1[in1_off + i + 3];
      auto v_4 = in_mem0[in0_off + i + 4] + in_mem1[in1_off + i + 4];
      auto v_5 = in_mem0[in0_off + i + 5] + in_mem1[in1_off + i + 5];
      auto v_6 = in_mem0[in0_off + i + 6] + in_mem1[in1_off + i + 6];
      auto v_7 = in_mem0[in0_off + i + 7] + in_mem1[in1_off + i + 7];
      out_mem[out_off + i + 0] += v_0;
      out_mem[out_off + i + 1] += v_1;
      out_mem[out_off + i + 2] += v_2;
      out_mem[out_off + i + 3] += v_3;
      out_mem[out_off + i + 4] += v_4;
      out_mem[out_off + i + 5] += v_5;
      out_mem[out_off + i + 6] += v_6;
      out_mem[out_off + i + 7] += v_7;
    }
    for (; i < N; i++) {
      out_mem[out_off + i + 0] +=
          in_mem0[in1_off + i + 0] + in_mem1[in1_off + i + 0];
    }
  };
};

bool condition(const LoopTree &lt, LoopTree::TreeRef ref) {
#define REQ(cond)                                                              \
  if (!(cond)) {                                                               \
    return false;                                                              \
  }

  REQ(lt.node(ref).kind == LoopTree::LOOP);
  REQ(lt.node(ref).children.size() == 1);

  auto c = lt.node(ref).children.at(0);
  REQ(lt.node(c).kind == LoopTree::NODE);

  // contiguity check
  const auto &node = lt.ir.node(lt.node(c).node);
  const auto &vars = node.vars();
  REQ(vars.back() == lt.node(ref).loop.var);
#undef REQ
  return true;
}

InnerFnType unroll_contiguous(const LoopTree &lt, const Auxiliary &aux,
                              LoopTree::TreeRef ref) {
  if (!condition(lt, ref)) {
    return InnerFnType{};
  }
  auto c = lt.node(ref).children.at(0);
  const auto &node = lt.ir.node(lt.node(c).node);
  const auto &vars = node.vars();

  auto N = lt.node(ref).loop.size;
  if (lt.ir.node(lt.node(c).node).op() == "add") {
    return gen_vec_add(lt, aux.allocs, c, N);
  } else if (lt.ir.node(lt.node(c).node).op() == "read") {
    return gen_vec_read(lt, c, aux.allocs.at(lt.node(c).node), N);
  } else if (lt.ir.node(lt.node(c).node).op() == "write") {
    return gen_vec_write(lt, aux.allocs, c, N);
  }

  return InnerFnType{};
}

bool unroll_contiguous_aux(const LoopTree &lt, Auxiliary &aux,
                           LoopTree::TreeRef ref) {
  return false;
  if (!condition(lt, ref)) {
    return false;
  }
  auto c = lt.node(ref).children.at(0);
  auto N = lt.node(ref).loop.size;
  if (lt.ir.node(lt.node(c).node).op() == "add") {
    gen_alloc(lt, aux, c);
    return true;
  } else if (lt.ir.node(lt.node(c).node).op() == "read") {
    gen_alloc(lt, aux, c);
    return true;
  } else if (lt.ir.node(lt.node(c).node).op() == "write") {
    gen_alloc(lt, aux, c);
    return true;
  }
  return false;
}

// REGISTER_BACKEND(generic, unroll_contiguous, unroll_contiguous_aux);
