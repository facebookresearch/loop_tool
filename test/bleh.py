import loop_tool as lt
import pdb
pdb.set_trace()
a = lt.Tensor(128,128)
b = lt.Tensor(128,128)
m, n, k = lt.symbols("m n k")

with lt.Backend("loop_nest"):
  c = (a.to(m, k) * b.to(k, n)).sum(k)
  lt.ui(c)
