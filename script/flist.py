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
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')