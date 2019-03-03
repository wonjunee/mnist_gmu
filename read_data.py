tf = []
mat2 = []
mat4 = []
mat6 = []

with open("output_without_batch.txt") as F:
	for i in F:

		if "EPOCHS" in i:
			print()
			print(i)

		if "Test Accuracy" in i:
			tf.append(float(i.split(" ")[-1]))
			print("TF accuracy:", float(i.split(" ")[-1]))

		if "train_mat_type: 2" in i:
			mat2.append(float(i.split(" ")[-1]))
			print("mat 2 accuracy:", float(i.split(" ")[-1]))

		if "train_mat_type: 4" in i:
			mat4.append(float(i.split(" ")[-1]))
			print("mat 4 accuracy:", float(i.split(" ")[-1]))

		if "train_mat_type: 6" in i:
			mat6.append(float(i.split(" ")[-1]))
			print("mat 6 accuracy:", float(i.split(" ")[-1]))


import pickle
print(len(tf))
print(len(mat2))
print(len(mat4))
print(len(mat6))

pickle.dump({"tf":tf[:len(mat6)], "mat2":mat2[:len(mat6)], "mat4":mat4[:len(mat6)], "mat6":mat6},open("without_batch.p","wb"))


a = pickle.load(open("without_batch.p", "rb"))

tf = a["tf"]
mat2 = a["mat2"]
mat4 = a["mat4"]
mat6 = a["mat6"]

import matplotlib.pyplot as plt
x = range(1, len(tf)+1)
plt.figure(figsize=(10,5))
plt.plot(x, tf, 'o-', label="tf")
plt.plot(x, mat2, 'o-', label="mat2")
plt.plot(x, mat4, 'o-', label="mat4")
plt.plot(x, mat6, 'o-', label="mat6")
plt.legend()
plt.grid()
plt.title("Training without batch")
plt.savefig("Training without batch.png")
plt.show()