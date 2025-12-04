import numpy as np
import imageio
import os

images = np.load('output/camera_images.npy')
images_rgb = images[:, :, :, :3]
images_rgb = images_rgb.astype(np.uint8)
print(f"Images RGB shape: {images_rgb.shape}")

# save images to a video
imageio.mimsave('output/camera_images.mp4', images_rgb, fps=1)
print(f"Saved images to output/camera_images.mp4")