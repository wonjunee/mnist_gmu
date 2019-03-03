import pickle

a = pickle.load(open("with_batch.p", "rb"))
b = pickle.load(open("without_batch.p", "rb"))

tf_with = a["tf"]
mat2_with = a["mat2"]
mat4_with = a["mat4"]
mat6_with = a["mat6"]

tf_without = b["tf"]
mat2_without = b["mat2"]
mat4_without = b["mat4"]
mat6_without = b["mat6"]

import matplotlib.pyplot as plt

x = range(1, len(tf_with)+1)
y = range(1, len(tf_without)+1)
plt.figure(figsize=(10,5))
plt.plot(x, tf_with, 'o-', label="tf with batch")
plt.plot(x, mat2_with, 'o-', label="mat2 with batch")
plt.plot(x, mat4_with, 'o-', label="mat4 with batch")
plt.plot(x, mat6_with, 'o-', label="mat6 with batch")
plt.plot(y, tf_without, 'o-', label="tf without batch")
plt.plot(y, mat2_without, 'o-', label="mat2 without batch")
plt.plot(y, mat4_without, 'o-', label="mat4 without batch")
plt.plot(y, mat6_without, 'o-', label="mat6 without batch")
plt.legend()
plt.grid()
plt.title("Training with without batch")
plt.savefig("Training with without batch.png")
plt.show()

