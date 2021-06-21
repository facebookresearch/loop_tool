## Regs, Threads -> Vectors, Warps, Blocks

1. Loops dictate memory usage
- scope based

```
alloc T1(a, b)
for a:
  for b:
    T1[a,b] = read(a,b)
for b:
  for a:
    T2[b] += T1[a,b]
```

vs

```
for a:
  alloc T1(b)
  for b:
    T1[b] = read(a,b)
  for b:
    T2[b] += T1[b]
```

```
// warp based
for i in log(a):
  val += shufledown(val, i)
// sync based
__sync
if thread(a) == 0:
  for a:
    val += vals[a] 
// atomic based
val = atomicAdd(val, thread(a))

``` 

2. Parallelism of loops dictates memory usage in a similar way
- "reverse scope based"


- if its going to be reduced, all inner parallel dims are multiplied
  - either followed by a reduce on a single thread (compute crosses warps)
  - or a warp reduce (compute within warps)
- if its going to be pointwise, all inner parallel dims are dividied

```
for a:
  alloc T1(2)
  for b in 8: // parallel
    b' = b / 4 // calc from other parallel
    T1[b'] = read(a,b)
  for b in 2: // not parallel
    for b' in 4: // parallel
      T2[b] += T1[]
```

```
alloc shared T1[2]
for a in 8: // not parallel
  for a' in 2: // parallel
    T1[a'] += read()
__syncthreads()
if tid == 0:
  for a' in 2: // not parallel
    T2[] += T1[a']
```


