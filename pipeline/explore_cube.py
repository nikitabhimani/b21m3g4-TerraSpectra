import numpy as np

# Fake hyperspectral cube: (height, width, bands)
cube = np.random.rand(64, 64, 200).astype(np.float32)

print("Cube shape:", cube.shape)
print("Number of spectral bands:", cube.shape[2])
print("Spectrum of pixel (10, 10) - first 5 bands:", cube[10, 10, :5])

# Reshape to (pixels, bands): the format PCA needs
pixels = cube.reshape(-1, cube.shape[2])
print("Reshaped for PCA:", pixels.shape)