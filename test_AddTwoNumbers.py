import mlx.core as mx
from mlx_graphs_extension import addTwoNumbers

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([1.4, 5.6, 7.8])
c = addTwoNumbers(a, b, stream=mx.cpu)

print(f"c shape: {c.shape}")
print(f"c dtype: {c.dtype}")
print(c)
print(f"c correct: {mx.all(c == 6.0).item()}")