import loop_tool_py as lt
import nn

import math
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
from extra.utils import fetch, fake_torch_load, get_child

class MBConvBlock:
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.uniform(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2D(oup)
    else:
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      self.pad = [(kernel_size-1)//2]*4

    self._depthwise_conv = Tensor.uniform(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2D(oup)

    self.has_se = has_se
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = Tensor.uniform(num_squeezed_channels, oup, 1, 1)
      self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
      self._se_expand = Tensor.uniform(oup, num_squeezed_channels, 1, 1)
      self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.uniform(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2D(output_filters)

  def __call__(self, inputs):
    x = inputs
    if self._expand_conv:
      x = self._bn0(x.conv2d(self._expand_conv)).swish()
    x = x.pad2d(padding=self.pad)
    x = x.conv2d(self._depthwise_conv, stride=self.strides, groups=self._depthwise_conv.shape[0])
    print(self.strides)
    return x
    x = self._bn1(x).swish()

    if self.has_se:
      x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])
      x_squeezed = x_squeezed.conv2d(self._se_reduce, self._se_reduce_bias).swish()
      x_squeezed = x_squeezed.conv2d(self._se_expand, self._se_expand_bias)
      x = x.mul(x_squeezed.sigmoid())

    x = self._bn2(x.conv2d(self._project_conv))
    #x = x.conv2d(self._project_conv)
    #return x
    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x

class EfficientNet:
  def __init__(self, number=0, classes=1000, has_se=True):
    self.number = number
    global_params = [
      # width, depth
      (1.0, 1.0), # b0
      (1.0, 1.1), # b1
      (1.1, 1.2), # b2
      (1.2, 1.4), # b3
      (1.4, 1.8), # b4
      (1.6, 2.2), # b5
      (1.8, 2.6), # b6
      (2.0, 3.1), # b7
      (2.2, 3.6), # b8
      (4.3, 5.3), # l2
    ][number]

    def round_filters(filters):
      multiplier = global_params[0]
      divisor = 8
      filters *= multiplier
      new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))

    out_channels = round_filters(32)
    self._conv_stem = Tensor.uniform(out_channels, 3, 3, 3)
    self._bn0 = BatchNorm2D(out_channels)
    blocks_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]
    # num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio

    self._blocks = []
    for b in blocks_args:
      args = b[1:]
      args[3] = round_filters(args[3])
      args[4] = round_filters(args[4])
      for n in range(round_repeats(b[0])):
        #print(', '.join(str(a) for a in args))
        self._blocks.append(MBConvBlock(*args, has_se=has_se))
        args[3] = args[4]
        args[1] = (1,1)

    in_channels = round_filters(320)
    out_channels = round_filters(1280)
    self._conv_head = Tensor.uniform(out_channels, in_channels, 1, 1)
    self._bn1 = BatchNorm2D(out_channels)
    self._fc = Tensor.uniform(out_channels, classes)
    self._fc_bias = Tensor.zeros(classes)

  def forward(self, x):
    x = x.pad2d(padding=(0,1,0,1))
    x = self._bn0(x.conv2d(self._conv_stem, stride=2)).swish()
    x = x.sequential(self._blocks)
    x = self._bn1(x.conv2d(self._conv_head)).swish()
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, x.shape[1]))
    return x.linear(self._fc, self._fc_bias)

  def load_from_pretrained(self):
    model_urls = {
      0: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
      1: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
      2: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
      3: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
      4: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
      5: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
      6: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
      7: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
    }


    b0 = fake_torch_load(fetch(model_urls[self.number]))
    for k,v in b0.items():
      for cat in ['_conv_head', '_conv_stem', '_depthwise_conv', '_expand_conv', '_fc', '_project_conv', '_se_reduce', '_se_expand']:
        if cat in k:
          k = k.replace('.bias', '_bias')
          k = k.replace('.weight', '')

      #print(k, v.shape)
      mv = get_child(self, k)
      vnp = v.astype(np.float32)
      vnp = vnp if k != '_fc' else vnp.T
      vnp = vnp if vnp.shape != () else np.array([vnp])

      if mv.shape == vnp.shape:
        mv.assign(vnp)
      else:
        print("MISMATCH SHAPE IN %s, %r %r" % (k, mv.shape, vnp.shape))



import numpy as np
#arrays = np.load("efficient_net_0.npz")
#for f in arrays.files:
#  if "_bn0" in f:
#    print(f, arrays[f].flatten()[0])
#m = EfficientNet(number=0)
args = [
  [3, (1, 1), 1, 32, 16, 0.25],
  [3, (2, 2), 6, 16, 24, 0.25],
  [3, (1, 1), 6, 24, 24, 0.25],
  [5, (2, 2), 6, 24, 40, 0.25],
  [5, (1, 1), 6, 40, 40, 0.25],
  [3, (2, 2), 6, 40, 80, 0.25],
  [3, (1, 1), 6, 80, 80, 0.25],
  [3, (1, 1), 6, 80, 80, 0.25],
  [5, (1, 1), 6, 80, 112, 0.25],
  [5, (1, 1), 6, 112, 112, 0.25],
  [5, (1, 1), 6, 112, 112, 0.25],
  [5, (2, 2), 6, 112, 192, 0.25],
  [5, (1, 1), 6, 192, 192, 0.25],
  [5, (1, 1), 6, 192, 192, 0.25],
  [5, (1, 1), 6, 192, 192, 0.25],
  [3, (1, 1), 6, 192, 320, 0.25],
]

def MBConv(inp, exp_conv, exp_bn, dw_conv, dw_stride, dw_bn, sq, sq_bias, ex, ex_bias, proj_conv, proj_bn):
  X = inp
  #print("input", X.symbolic_shape)
  if exp_conv:
    X = nn.linear(X, exp_conv)
    #X = nn.conv(X, exp_conv, X.symbolic_shape[:2], exp_conv.symbolic_shape[:2])
    b, y, x, c = X.symbolic_shape
    X = nn.swish(nn.batch_norm(X, *exp_bn)).transpose(b, c, y, x)
  else:
    shape = X.symbolic_shape
    shape[1] = dw_conv.symbolic_shape[0]
    X = X.to(*shape)
  spatial = X.symbolic_shape[-2:]
  pad = (dw_conv.shape[-1] - 1) // 2
  if dw_stride == 2:
     pad = (pad-1, pad)
  #print('before', X.symbolic_shape)#, dw_conv.symbolic_shape)
  #print('before', X.symbolic_shape)
  X = nn.pad(X, *[(s, pad) for s in spatial])
  spatial = X.symbolic_shape[-2:]
  print(spatial, dw_stride)
  X = nn.conv(X, dw_conv, spatial, dw_conv.symbolic_shape[-2:], stride=dw_stride, channel_reduce=False)
  return X
  #print('after', X.symbolic_shape)
  #print('after', X.symbolic_shape, dw_conv.symbolic_shape)
  #print(X.shape)
  X = nn.swish(nn.batch_norm(X, *dw_bn))
  #X = nn.batch_norm(X, *dw_bn)
  #return nn.swish(X)#.transpose(a, d, b, c)
  #print("first conv", X.symbolic_shape)
  #X = nn.avg_pool(X, X.symbolic_shape[2:4])
  #X = nn.linear(X, sq, sq_bias)
  #X = nn.linear(X, exp, exp_bias)
  X = nn.linear(X, proj_conv)
  #print(X.shape, X.symbolic_shape)#, dw_conv.symbolic_shape)
  #a, b, c, d = X.symbolic_shape
  #print(X.symbolic_shape, proj_bn[0].symbolic_shape)
  #return X.transpose(a, d, b, c)
  #######X = nn.batch_norm(X, *proj_bn)
  #print(X.symbolic_shape)
  #print("projected", X.symbolic_shape, inp.symbolic_shape)
  if X.symbolic_shape == inp.symbolic_shape:
    X = X + inp
  b, y, x, c = X.symbolic_shape
  return X.transpose(b, c, y, x)

np.random.seed(123)
#mb._depthwise_conv.data = np.ones(mb._depthwise_conv.data.shape)
#mb._project_conv.data = np.ones(mb._project_conv.data.shape)
#mb._bn2.running_mean.data = np.zeros(mb._bn2.running_mean.data.shape)
#mb._bn2.running_var.data = np.ones(mb._bn2.running_var.data.shape)
#mb._bn2.weight.data = np.ones(mb._bn2.weight.data.shape)
#mb._bn2.bias.data = np.zeros(mb._bn2.bias.data.shape)
#print(mb._bn2.running_mean.data.shape)
#print(mb._bn2.running_var.data.shape)

for i in range(1, len(args)):
    mb = MBConvBlock(*args[i], False)
    X = np.random.randn(1, args[i][3], 8, 8)
    #X = np.ones((1, args[i][3], 8, 8))
    X_tg = Tensor(X)
    Y_tg = mb(X_tg)

    s = lt.SymbolGenerator()
    X_lt = lt.Tensor(s.B, s.C, s.Y, s.X).set(X)
    dw = lt.Tensor(s.Ic, s.Ky, s.Kx).set(mb._depthwise_conv.data.squeeze())
    dw_bn = [lt.Tensor(s.Ic) for _ in range(4)]
    dw_bn[0].set(mb._bn1.running_mean.data)
    dw_bn[1].set(mb._bn1.running_var.data)
    dw_bn[2].set(mb._bn1.weight.data)
    dw_bn[3].set(mb._bn1.bias.data)
    proj = lt.Tensor(s.Oc, s.Ic).set(mb._project_conv.data.squeeze())
    proj_bn = [lt.Tensor(s.Oc) for _ in range(4)]
    proj_bn[0].set(mb._bn2.running_mean.data)
    proj_bn[1].set(mb._bn2.running_var.data)
    proj_bn[2].set(mb._bn2.weight.data)
    proj_bn[3].set(mb._bn2.bias.data)
    exp_factor = args[i][2]
    exp_conv = None
    exp_bn = None
    if exp_factor > 1:
        exp_conv = lt.Tensor(s.Ic, s.C).set(mb._expand_conv.data.squeeze())
        exp_bn = [lt.Tensor(s.Ic) for _ in range(4)]
        exp_bn[0].set(mb._bn0.running_mean.data)
        exp_bn[1].set(mb._bn0.running_var.data)
        exp_bn[2].set(mb._bn0.weight.data)
        exp_bn[3].set(mb._bn0.bias.data)
    Y_lt = MBConv(X_lt, exp_conv, exp_bn, dw, args[i][1][0], dw_bn, None, None, None, None, proj, proj_bn)
    #print(Y_lt.symbolic_shape)
    print("jjjj", Y_tg.data.flatten()[0:3], Y_lt.numpy().flatten()[0:3])
    print(Y_tg.shape, Y_lt.shape)
    assert np.allclose(Y_tg.data, Y_lt.numpy(), rtol=0.001, atol=0.001)
    print(f"{i} passed")
    if i == 1:
        break
