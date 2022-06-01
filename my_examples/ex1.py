import loop_tool_py as lt
import numpy as np
import pdb


print(lt.backends())

def mm(a, b):
  M, N, K = lt.Symbol("M"), lt.Symbol("N"), lt.Symbol("K")
  c = a.to(M, K) * b.to(K, N)
  return c.sum(K)

x_np = np.random.randn(128,128)
y_np = np.random.randn(128,128)

x = lt.Tensor(x_np)
y = lt.Tensor(y_np)


# x = lt.Tensor(128,128)
# y = lt.Tensor(128,128)

# x.set(np.random.randn(128,128))
# y.set(np.random.randn(128,128))

j
pdb.set_trace()

z = mm(x, y)
z_ref = x.numpy() @ y.numpy()
print("max error", np.max(np.abs(z.numpy() - z_ref)))
