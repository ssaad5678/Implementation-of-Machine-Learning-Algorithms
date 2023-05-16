import numpy as np

X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float) / 9
y = np.array([[92], [86], [89]], dtype=float) / 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

np.random.seed(42)
wh = np.random.uniform(size=(2, 3))
bh = np.random.uniform(size=(1, 3))
wout = np.random.uniform(size=(3, 1))
bout = np.random.uniform(size=(1, 1))

for i in range(5000):
    hlayer_act = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)

    d_output = (y - output) * derivatives_sigmoid(output)
    d_hiddenlayer = np.dot(d_output, wout.T) * derivatives_sigmoid(hlayer_act)

    wout += np.dot(hlayer_act.T, d_output) * 0.1
    bout += np.sum(d_output, axis=0, keepdims=True) * 0.1
    wh += np.dot(X.T, d_hiddenlayer) * 0.1
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * 0.1

    if i % 1000 == 0:
        print("Iteration:", i)
        print("Output:", output)

print("Input:\n", X.astype(int))
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
