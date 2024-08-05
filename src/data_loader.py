import os
import numpy as np
from PIL import Image
import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

def load_dataset(image_dir, label_dir, target_size=(224, 224)):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            img_array = load_and_preprocess_image(img_path, target_size)
            
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
            
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)