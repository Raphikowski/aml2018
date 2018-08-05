"""Create folders with images from fer-csv and ferplus-csv."""

import os
import time
import pickle
import numpy as np
from PIL import Image

# Output will look like this:
#   base/Training/fer00001.png ...
#   base/PublicTest/ferxxxxx.png ...
#   base/PrivateTest/ferxxxxx.png ...
#   base/fer_Training.pkl ...

# fer_{}.pkl - list of [fer_name, ferplus_name, emotion, votes]
#   fer_name: ferXXXXX.png
#   ferplus_name: ferXXXXXXXX.png (images of class NF are left out, therefore different from fer_name)
#   emotion: class int
#   votes: list of ferplus votes on classes, sum(votes) = 10

# NOTE: Customize paths if necessary (base is the output directory)
base = '/Users/lennard/data/project/fer'
fercsv = '/Users/lennard/data/project/fer2013.csv'
ferpluscsv = '/Users/lennard/data/project/fer2013new.csv'

if not os.path.exists(base):
        os.makedirs(base)

# Open csv files
with open(fercsv, 'r') as f:
    fer = f.readlines()
with open(ferpluscsv, 'r') as f:
    ferplus = f.readlines()

# Initialize labels
# NOTE: images is only needed to calculate mean-images and std-images of the data
labels = {'PrivateTest': [], 'PublicTest': [], 'Training': []}
images = {'PrivateTest': [], 'PublicTest': [], 'Training': []}

print('\nCreating images...')

# Create images
t0 = time.time()
for i, (fer_line, ferplus_line)  in enumerate(zip(fer[1:], ferplus[1:])):
    
    # Read data from both datasets
    fer_name = f'fer{i:05}.png'
    emotion, pixels, fer_usage = fer_line.strip().split(',')
    ferplus_usage, ferplus_name, *votes = ferplus_line.strip().split(',')
    assert fer_usage == ferplus_usage
    
    # Save image
    array = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
    images[fer_usage].append(array)
    image = Image.fromarray(array)
    
    folder = os.path.join(base, fer_usage)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    image.save(os.path.join(folder, fer_name), compress_level=0)
    
    # Write label information into file
    votes = [int(vote) for vote in votes]
    labels[fer_usage].append([fer_name, ferplus_name, int(emotion), votes])
    t = int(time.time() - t0)
    print(f't={t:03}s: {fer_name}', end='\r')
    

print('\n\nCreating pickle files...')

# Calculate mean and std of "Training" for normalization
means = {}
stds = {}
for split in images.keys():
    array = np.array(images[split], dtype=np.float64)
    means[split] = array.mean(axis=0)
    stds[split] = array.std(axis=0)
    
with open(os.path.join(base, 'mean_Training.pkl'), 'wb') as f:
    pickle.dump(means['Training'], f)
    
with open(os.path.join(base, 'std_Training.pkl'), 'wb') as f:
    pickle.dump(stds['Training'], f)


# Save list of labels and names into base directory
# fer and ferplus differ because images of the class NotAFace are excluded from ferplus
for split in labels.keys():
    fer_pkl = []
    ferplus_pkl = []
    for label in labels[split]:
        name, plus_name, emotion, votes = label
        # Not a face
        fer_pkl.append([name, emotion, votes])
        if label[1] == '':
            continue
        ferplus_pkl.append([name, emotion, votes])

    with open(os.path.join(base, f'fer_{split}.pkl'), 'wb') as f:
        pickle.dump(fer_pkl, f)

    with open(os.path.join(base, f'ferplus_{split}.pkl'), 'wb') as f:
        pickle.dump(ferplus_pkl, f)

print('\nDone')
