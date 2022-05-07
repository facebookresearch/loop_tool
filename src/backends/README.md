# Adding a Backend to `loop_tool`

This document is a guide to adding a backend to `loop_tool`.
The basic premise is that `loop_tool` has a very useful `LoopTree` structure
and backends can lower that to an object that has a `run()` function.

There is a class `Compiler` that makes the lowering process *much* easier.
In fact, `Compiler` is so useful that the entire `cpp` backend is ~500 lines.

## The Simplest Integration Example

The minimal integration example is a dozen lines and can be found at the top
of `test/test_backend.cpp`.  This does not do anything useful and is largely
a guide of general usage.

A more involved (non-trivial) example can be found
in `src/backends/cpu/cpp.cpp`.
Ideas present in this example are discussed in the next section:
**Explanation of the IR and Lowering**.


#### `loop_tool::Backend` Structure

A backend implements two functions:

```cpp
std::unique_ptr<Compiled> compile_impl(const LoopTree &lt) const;
```
This function returns an instance of a `Compiled` object (explained later).
Some frontends (such as `lazy`) will never
call `compile_impl` twice for the same `LoopTree`.

```cpp
int hardware_requirement() const;
```

This function returns an indicator of what hardware is required to run
this backend.  This enables runtime registration.

Here's a full example of a `CustomBackend`.  The `CustomCompiled` is explained later.

```cpp
using namespace loop_tool;

struct CustomBackend : Backend {
  CustomBackend() : Backend("custom") {}

  std::unique_ptr<Compiled> compile_impl(const LoopTree &lt) {
    return std::make_unique<CustomCompiled>();
  }

  int hardware_requirement() const {
    return 0;  // CPU
  }
};
```

#### Registration

Once you have a backend created, you'll need to register it.
That can be done by declaring a static variable of type `RegisterBackend`.

This is the only boilerplate and will make the backend discoverable in
every frontend (C++, Python, JavaScript).

```
using namespace loop_tool;

static RegisterBackend reg_{std::make_shared<CustomBackend>()};
```

#### `loop_tool::Compiled` Structure

A `Compiled` object in `loop_tool` implements two functions:

```cpp
void run(const std::vector<void *> &memory, bool sync = true) const;
```
and optionally
```cpp
std::string dump() const;
```
for debugging.

Here's a full example:

```cpp
struct CustomCompiled : public Compiled {
  void run(const std::vector<void *> &memory, bool sync) const override {
    std::cerr << "here!\n";
    return;
  }
  
  std::string dump() const {
  	return "TODO";
  }
};
```

Note that this does very little and certainly wouldn't be useful
in any context beyond demonstration :)



# Explanation of the IR and Lowering

To do something useful in the `run` function, we'll need to review
the core structures of `loop_tool`.
The primary structure (which can be serialized and deserialized without
loss of information), is the annotated dataflow graph: `loop_tool:IR`.

This dataflow graph is automatically
lowered to a more useful `loop_tool::LoopTree` class when used
with a backend for execution.

Optionally, the `loop_tool::LoopTree` can be analyzed by the
`loop_tool::Compiler` class, which calculates things such as
minimal allocation sizes, symbolic indexing expressions, and access constraints
(for valid indexing in the case of padding or concatenation).

## Using the `LoopTree` Class

`LoopTree`s are an exact representation of the control flow of a `loop_tool`
program.  Instead of control flow graphs, they are control flow trees.
To pretty-print a `LoopTree`, simply call `dump`:

```cpp
LoopTree lt; // assume this is populated
std::cerr << lt.dump();
```

As such, they are composed of "loops" (non-leaf nodes) and
"computational nodes" (leaf nodes).  Every abstract node in the `LoopTree`
is addressed by a `LoopTree::TreeRef` (which are just 32-bit pointers).

We can traverse a `LoopTree` as one would a normal tree.

```cpp
LoopTree::TreeRef ref = lt.roots.at(0);
std::vector<LoopTree::TreeRef> children = lt.children(ref);
auto ref_ = lt.parent(children.at(0));
ASSERT(ref_ == ref);
```

We can inspect `LoopTree::TreeRef` types to see if they are loops or
computational nodes:

```cpp
auto k = lt.kind(ref);
if (k == LoopTree::LOOP) {
  emit_loop(lt, ref);
} else if (k == LoopTree::NODE) {
  emit_compute(lt, ref);
}
```

To dereference the `LoopTree::TreeRef`s and access the underlying data:

```cpp
if (k == LoopTree::LOOP) {
  auto loop = lt.loop(ref);
  std::cerr << loop.size << " tail " << loop.tail << "\n";
} else if (k == LoopTree::NODE) {
  auto node_ref = lt.node(ref);
  // lt.node returns a 32-bit pointer into an IR structure
  const auto& node = lt.ir.node(node_ref);
  if (node.op() == Operation::multiply) {
  	std::cerr << "multiply!\n";
  } else {
    std::cerr << "not multiply\n";
  }
}
```

It's important to note that the `LoopTree` is a control flow tree
that references compute nodes in the data-flow graph.
A further dereference from the DFG (called `IR`) is required to
access the `IR::Node` objects.

#### `IR::Node`

`IR::Node`s work as expected of a DFG, containing only information
about inputs and outputs.  However, since the IR is focused
on virtual n-dimensional arrays,
we also have information about variables in each node.

```cpp
const auto& node = lt.ir.node(node_ref);
// 32-bit pointers are preserved across copies of IR or LoopTree
auto input_ref = node.inputs().at(0);
const auto& input = lt.ir.node(input_ref);
// more pointers
std::vector<IR::VarRef> vars = input.vars();
auto& v = lt.ir.var(vars.at(0));
std::cerr << "var name: " << v.name() << "\n";
```

The `IR::VarRef`s returned by `IR::Node::vars()` are the same `IR::VarRef`s
that would be found in the loops in a `LoopTree` (`loop.var`).

#### `LoopTree::walk`

Another useful operation in `LoopTree` is `walk`

```cpp
lt.walk([&](LoopTree::TreeRef ref, int depth) {
  // do something
});
```
which traverses the tree in a depth-first order (with respect to the tree)
and topologically ordered (with respect to the `IR`).

## Using the `Compiler` Class

The `Compiler` class has a number of useful constructs.

#### `Compiler::Allocation`

A map of `IR::NodeRef` (from the DFG) to `Compiler::Allocation`
is calculated upon construction.

```cpp
std::unordered_map<IR::NodeRef, Allocation> allocations;
```
An `Allocation` contains

- `mem_idx`: the index into the `void* memory` passed into the `run`.
- `int64_t size() const`: the size of the memory allocation.
- `NodeRef lca`: the least common ancestor of the allocation and all dependent compute nodes.

#### `Compiler::Access`

A function to help with determing how to access input memory is

```cpp
Access gen_access(IR::NodeRef node, LoopTree::TreeRef ref) const;
```

The function can be used to determine how DFG node `node_ref` should be access
from control flow point `ref` (either as an input or output).
This is important because some memory can be iterated over in multiple ways
and index semantic calculation can become extremely tricky.

```cpp
auto access = gen_access(node_ref, ref);
const auto& idx_expr = get_scoped_expr(access);
std::cerr << idx_expr.dump() << "\n";
```

The `Access` can also be used to derive constraints with

```cpp
std::vector<std::pair<symbolic::Expr, int64_t>> get_constraints(const Access &access) const;
```

A vector of symbolic expressions and their maximum values (minimum is implicitly zero).
This makes it possible to determine if the access *should be accessed*
at a given location.  If padding is used, for example, there are
index expressions that should map to zero and would be represented with
one of these constraints.

#### `symbolic::Sym` <-> `IR::VarRef`

To simplify implementations,
there is a provided symbolic expression library in `loop_tool`.
However, when analyzing DFG nodes, the associated `IR:VarRef`s
often need to be mapped to the symbolic expressions passed around `Compiler`
and vice versa.

To do this, `Compiler` provides two maps (calculated at construction time):

```cpp
std::unordered_map<IR::VarRef, symbolic::Symbol> var_to_sym;
std::unordered_map<symbolic::Symbol, IR::VarRef> sym_to_var;
```

This makes it straight-forward to work in whichever domain is more convenient.

```cpp
IR::VarRef v; // assume populated
Expr expr; // assume initialized

if (loop.var == v) {
  auto sym = var_to_sym.at(v);
  auto stride_expr = differentiate(expr, sym);
  auto stride_for_float = (stride_expr * 4).evaluate();
}
```

## Code Pointers

There are two interesting uses of the `Compiler` class today (with more on the way).

The WebAssembly backend, which uses an assembler to generate a binary,
can be found in `src/backends/wasm/wasm.cpp`.

The C++ codegen backend, which generates a valid C++ text program,
can be found in `src/backends/cpu/cpp.cpp`.

