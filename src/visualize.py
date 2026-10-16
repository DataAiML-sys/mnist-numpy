import numpy as np
import matplotlib.pyplot as plt

with open("data/train-images-idx3-ubyte", 'rb') as f:
    f.read(16)
    images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(60000, 28, 28)

with open("data/train-labels-idx1-ubyte", 'rb') as f:
    f.read(8)
    labels = np.frombuffer(f.read(), dtype=np.uint8)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap = 'gray')
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_digits.png", dpi = 100)
print("Saved sample_digits.png")
plt.show()