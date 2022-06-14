/*
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include <cassert>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "error.h"
#include "symbolic.h"

namespace loop_tool {

#define OPS(_)  \
  _(constant)   \
  _(write)      \
  _(read)       \
  _(view)       \
  _(copy)       \
  _(add)        \
  _(subtract)   \
  _(multiply)   \
  _(divide)     \
  _(min)        \
  _(max)        \
  _(log)        \
  _(exp)        \
  _(sqrt)       \
  _(abs)        \
  _(negate)     \
  _(reciprocal) \
  _(name)

// Operations on arrays
enum struct Operation {
#define X(op) op,
  OPS(X)
#undef X
};

inline std::string dump(const Operation &op) {
#define X(o)         \
  case Operation::o: \
    return #o;
  switch (op) {
    OPS(X)
    default:
      ASSERT(0) << "unkown op code " << (int)op;
      return "unknown";
  }
  return "unknown";
#undef X
}

template <typename T, template <typename _> class H = std::hash>
std::unordered_set<T, H<T>> to_set(std::vector<T> v) {
  std::unordered_set<T, H<T>> out;
  for (const auto &e : v) {
    out.insert(e);
  }
  return out;
}

template <typename T, template <typename _> class H = std::hash>
std::unordered_set<T, H<T>> intersection(const std::unordered_set<T, H<T>> &a,
                                         const std::unordered_set<T, H<T>> &b) {
  std::unordered_set<T, H<T>> c;
  for (const auto &e : a) {
    if (b.find(e) != b.end()) {
      c.insert(e);
    }
  }
  return c;
}

struct Node;
struct Var;

class IR {
 public:
  IR() {}
  // For code clarity
  using NodeRef = int32_t;
  using VarRef = int32_t;
  struct LoopSize {
    int64_t size;
    int64_t tail;
  };

  NodeRef create_node(Operation op, std::vector<NodeRef> inputs,
                      std::vector<VarRef> vars,
                      std::vector<symbolic::Constraint> constraints = {},
                      std::unordered_map<int, IR::VarRef> sym_var_map = {});
  VarRef create_var(std::string name);

  void delete_node(const NodeRef &node_ref);
  void replace_all_uses(NodeRef old_node, NodeRef new_node);
  void update_inputs(NodeRef node_ref, std::vector<NodeRef> inputs);
  void update_vars(NodeRef node_ref, std::vector<VarRef> vars);

  std::vector<VarRef> vars() const;
  std::vector<NodeRef> nodes() const;

  inline const Node &node(NodeRef ref) const {
    ASSERT(!deleted_.count(ref)) << "attempting to access deleted node";
    ASSERT(ref < nodes_.size()) << "node ref '" << ref << "' not valid";
    return nodes_[ref];
  }
  inline Node &node(NodeRef ref) {
    ASSERT(!deleted_.count(ref)) << "attempting to access deleted node";
    ASSERT(ref < nodes_.size()) << "node ref '" << ref << "' not valid";
    return nodes_[ref];
  }

  inline const Var &var(VarRef ref) const {
    ASSERT(ref < vars_.size()) << "var ref '" << ref << "' not valid";
    return vars_[ref];
  }
  inline Var &var(VarRef ref) {
    ASSERT(ref < vars_.size()) << "var ref '" << ref << "' not valid";
    return vars_[ref];
  }

  inline float priority(NodeRef ref) const { return priorities_[ref]; }
  inline const std::vector<std::pair<VarRef, LoopSize>> &order(
      NodeRef ref) const {
    return orders_[ref];
  }
  inline const std::unordered_set<int> not_reusable(NodeRef ref) const {
    return reuse_disabled_[ref];
  }

  inline const std::vector<NodeRef> &inputs() const { return inputs_; }
  inline const std::vector<NodeRef> &outputs() const { return outputs_; }

  inline void add_input(NodeRef input) { inputs_.emplace_back(input); }
  inline void add_output(NodeRef output) { outputs_.emplace_back(output); }
  inline void set_inputs(std::vector<NodeRef> inputs) {
    inputs_ = std::move(inputs);
  }
  inline void set_outputs(std::vector<NodeRef> outputs) {
    outputs_ = std::move(outputs);
  }

  // auxiliary information / annotations
  void reset_aux(IR::NodeRef node_ref);
  inline void set_priority(NodeRef ref, float priority) {
    priorities_[ref] = priority;
  }
  inline void set_order(NodeRef ref,
                        std::vector<std::pair<VarRef, LoopSize>> order) {
    // TODO validate order
    orders_[ref] = order;
    loop_annotations_[ref].clear();
    loop_annotations_[ref].resize(order.size());
  }
  inline void set_order(NodeRef ref,
                        std::vector<std::pair<VarRef, LoopSize>> order,
                        std::vector<std::string> annotations) {
    // TODO validate order
    orders_[ref] = order;
    ASSERT(annotations.size() == order.size());
    annotate_loops(ref, annotations);
  }
  inline void disable_reuse(NodeRef ref, int order_ref) {
    reuse_disabled_[ref].insert(order_ref);
  }
  inline void enable_reuse(NodeRef ref, int order_ref) {
    reuse_disabled_[ref].erase(order_ref);
  }

  // annotate a specific order index
  inline void annotate_loop(NodeRef ref, int idx, std::string annot) {
    loop_annotations_[ref].at(idx) = annot;
  }

  inline void annotate_loops(NodeRef ref, std::vector<std::string> annots) {
    loop_annotations_[ref] = annots;
  }

  inline std::vector<std::string> loop_annotations(NodeRef ref) const {
    return loop_annotations_.at(ref);
  }

  inline void annotate(NodeRef ref, std::string annot) {
    annotations_[ref] = annot;
  }

  inline std::string annotation(NodeRef ref) const {
    return annotations_.at(ref);
  }

  std::string dump(NodeRef ref) const;
  std::vector<VarRef> pointwise_vars(NodeRef ref) const;
  std::vector<VarRef> reduction_vars(NodeRef ref) const;
  std::vector<VarRef> view_reduction_vars(NodeRef idx) const;
  std::vector<VarRef> loop_vars(NodeRef ref) const;
  std::vector<VarRef> all_vars(NodeRef ref) const;
  void reify_deletions();

 private:
  std::vector<Node> nodes_;
  std::unordered_set<NodeRef> deleted_;
  // TODO consider efficient storage for splits/merges
  std::vector<Var> vars_;
  std::vector<float> priorities_;
  std::vector<std::vector<std::pair<VarRef, LoopSize>>> orders_;
  std::vector<std::unordered_set<int>> reuse_disabled_;
  std::vector<std::vector<std::string>> loop_annotations_;
  std::vector<std::string> annotations_;
  std::vector<NodeRef> inputs_;
  std::vector<NodeRef> outputs_;
};

std::string dot(const IR &ir);

class Node {
 protected:
  friend class IR;  // use the IR class to create nodes
  Node(Operation op, std::vector<IR::NodeRef> inputs,
       std::vector<IR::VarRef> vars,
       std::vector<symbolic::Constraint> constraints,
       std::unordered_map<int, IR::VarRef> sym_var_map)
      : op_(op),
        inputs_(inputs),
        vars_(vars),
        constraints_(constraints),
        sym_var_map_(sym_var_map) {}

  void replace_input(IR::NodeRef old_node, IR::NodeRef new_node);
  inline void update_inputs(std::vector<IR::NodeRef> inputs) {
    inputs_ = inputs;
  }
  inline void update_outputs(std::vector<IR::NodeRef> outputs) {
    outputs_ = outputs;
  }
  inline void update_vars(std::vector<IR::VarRef> vars) { vars_ = vars; }

 public:
  inline const std::vector<IR::NodeRef> &inputs() const { return inputs_; }
  inline const std::vector<IR::NodeRef> &outputs() const { return outputs_; }
  inline const std::vector<symbolic::Constraint> &constraints() const {
    return constraints_;
  }
  inline const IR::VarRef var(symbolic::Symbol sym) const {
    ASSERT(sym_var_map_.count(sym.id()))
        << "symbol " << sym.name() << "#" << sym.id()
        << " is not mapped to a variable";
    return sym_var_map_.at(sym.id());
  }
  inline bool has_sym(symbolic::Symbol sym) const {
    return sym_var_map_.count(sym.id());
  }
  inline const std::unordered_map<int, IR::VarRef> &sym_to_var() const {
    return sym_var_map_;
  }

  inline const Operation &op() const { return op_; }
  inline const std::vector<IR::VarRef> &vars() const { return vars_; }
  inline std::vector<IR::VarRef> &vars() { return vars_; }
  void remap_refs(const std::unordered_map<IR::NodeRef, IR::NodeRef> &);

 private:
  Operation op_;
  std::vector<IR::NodeRef> inputs_;
  // denote the output vars
  std::vector<IR::VarRef> vars_;
  // exclusively used for view operations
  std::vector<symbolic::Constraint> constraints_;
  std::unordered_map<int, IR::VarRef> sym_var_map_;

 protected:
  std::vector<IR::NodeRef> outputs_;
};

class Var {
 public:
  Var(std::string name, int version) : name_(name), version_(version) {}
  inline const std::string &name() const { return name_; }
  inline const int &version() const { return version_; }

 private:
  std::string name_;
  int version_;
};

// cheap, disposable data structures
struct LoopTree {
  using TreeRef = int;
  enum { NODE = 0, LOOP = 1 };

  struct Loop {
    IR::VarRef var;
    int var_depth;
    int64_t size;
    int64_t tail;
    bool operator==(const Loop &other) const {
      return var == other.var && var_depth == other.var_depth &&
             size == other.size && tail == other.tail;
    }
  };

  struct LoopTreeNode {
    TreeRef parent = -1;
    TreeRef idx = -1;
    int depth = 0;        // root depth
    int annotation = -1;  // index into LoopTree::annotations

    bool kind;  // 0 -> node, 1 -> loop
    union {
      IR::NodeRef node;
      Loop loop;
    };

    std::vector<int> children;

    LoopTreeNode(int p, int i, IR::NodeRef n)
        : parent(p), idx(i), node(n), kind(0) {}
    LoopTreeNode(int p, int i, const Loop &l)
        : parent(p), idx(i), loop(l), kind(1) {}
  };

  TreeRef add_leaf(TreeRef parent, IR::NodeRef n);
  TreeRef add_loop(TreeRef parent, const Loop &l);
  template <typename T>
  TreeRef add_node_impl(TreeRef parent, T n) {
    auto new_idx = nodes.size();
    nodes.emplace_back(parent, new_idx, n);
    if (parent == -1) {
      roots.emplace_back(new_idx);
      nodes[new_idx].depth = 0;
    } else {
      auto &parent_node = nodes[parent];
      nodes[new_idx].depth = parent_node.depth + 1;
      parent_node.children.emplace_back(new_idx);
    }
    return new_idx;
  }

  inline const LoopTreeNode &tree_node(TreeRef ref) const {
    ASSERT(ref < nodes.size());
    return nodes[ref];
  }
  inline LoopTreeNode &tree_node(TreeRef ref) {
    ASSERT(ref < nodes.size());
    return nodes[ref];
  }

  inline bool kind(TreeRef ref) const { return tree_node(ref).kind; }

  inline const IR::NodeRef node(TreeRef ref) const {
    ASSERT(kind(ref) == LoopTree::NODE);
    return tree_node(ref).node;
  }

  inline Loop loop(TreeRef ref) const {
    ASSERT(kind(ref) == LoopTree::LOOP);
    return tree_node(ref).loop;
  }

  std::unordered_set<IR::VarRef> scope_vars(TreeRef ref) const;

  inline const TreeRef parent(TreeRef ref) const {
    ASSERT(ref < nodes.size());
    return nodes[ref].parent;
  }

  inline const std::vector<TreeRef> &children(TreeRef ref) const {
    if (ref == -1) {
      return roots;
    }
    ASSERT(ref < nodes.size());
    return nodes[ref].children;
  }

  inline void annotate_(TreeRef ref, std::string annot) {
    for (auto i = 0; i < annotations.size(); ++i) {
      const auto &annotation = annotations[i];
      if (annot == annotation) {
        tree_node(ref).annotation = i;
        return;
      }
    }
    tree_node(ref).annotation = annotations.size();
    annotations.emplace_back(annot);
  }

  inline std::string annotation(TreeRef ref) const {
    auto annot_idx = tree_node(ref).annotation;
    if (annot_idx > -1) {
      return annotations[annot_idx];
    }
    return "";
  }

  inline int depth(TreeRef ref) const { return tree_node(ref).depth; }

  TreeRef lca(TreeRef a, TreeRef b) const;

  // like IR::order but includes variable versions
  std::vector<LoopTree::Loop> loop_order(IR::NodeRef ref) const;

  void walk(const std::function<void(LoopTree::TreeRef, int)> &fn,
            TreeRef start = -1) const;
  std::string dump(
      const std::function<std::string(LoopTree::TreeRef)> &fn = {}) const;
  IR ir;

  LoopTree(const IR &ir_);

  std::vector<LoopTreeNode> nodes;
  std::vector<TreeRef> roots;
  std::vector<std::string> annotations;
  std::unordered_map<IR::NodeRef, TreeRef> scheduled;
};

}  // namespace loop_tool
