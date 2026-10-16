import numpy as np

def load_images(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        print(f"  magic = {magic}, num = {num}, rows = {rows}, cols = {cols}")

        data = np.frombuffer(f.read(), dtype=np.uint8)

        data = data.reshape(num, rows, cols)

    return data

def load_labels(path):
    with open(path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        print(f"  magic = {magic}, num = {num}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

print("Loading training images...")
train_images = load_images("data/train-images-idx3-ubyte")

print("Loading training labels...")
train_labels = load_labels("data/train-labels-idx1-ubyte")

print("Loading test images...")
test_images = load_images("data/t10k-images-idx3-ubyte")

print("Loading test labels...")
test_labels = load_labels("data/t10k-labels-idx1-ubyte")

print()
print(f"Train images: {train_images.shape}  dtype={train_images.dtype}")
print(f"Train labels: {train_labels.shape}  dtype={train_labels.dtype}")
print(f"Test images:  {test_images.shape}   dtype={test_images.dtype}")
print(f"Test labels:  {test_labels.shape}   dtype={test_labels.dtype}")