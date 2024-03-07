import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='',help='')
parser.add_argument('--output', type=str, default='',help='')
args = parser.parse_args()

ext = {'.npy'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in tqdm(files):
        if os.path.splitext(file)[1] in ext:
            
            # if image is not noise, append
            img = np.load(os.path.join(root, file))
            img /= 255.
            
            if not(img.min() > 0.4 and img.max() < 0.6):
                images.append(os.path.join(root, file))

            del img

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')