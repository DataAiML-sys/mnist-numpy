import numpy as np

def load_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data

def load_labels(path):
    with open(path, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

train_images = load_images("data/train-images-idx3-ubyte")
train_labels = load_labels("data/train-labels-idx1-ubyte")
test_images = load_images("data/t10k-images-idx3-ubyte")
test_labels = load_labels("data/t10k-labels-idx1-ubyte")

X_train = train_images.reshape(-1, 784)
X_test = test_images.reshape(-1, 784)

print(f"Before flatten: {train_images.shape}")
print(f"After flatten:  {X_train.shape}")

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")

def one_hot(labels, num_classes=10):
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        result[i, label] = 1.0
    return result

Y_train = one_hot(train_labels)
Y_test = one_hot(test_labels)

print(f"Y_train shape: {Y_train.shape}")
print(f"Example: label={train_labels[0]}, one-hot={Y_train[0]}")

np.save("data/X_train.npy", X_train)
np.save("data/Y_train.npy", Y_train)
np.save("data/X_test.npy",  X_test)
np.save("data/Y_test.npy",  Y_test)

print("\nSaved prepared data to data/")