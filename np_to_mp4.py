import numpy as np
import imageio
import os
import glob
import natsort


# find all png files in the output directory and make a video of them
png_files = glob.glob('output/*.png')
png_files = sorted(png_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
images = [imageio.imread(png_file) for png_file in png_files]
imageio.mimsave('output/camera_images.mp4', images, fps=5)
print(f"Saved images to output/camera_images_long.mp4")