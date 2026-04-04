from PIL import Image
import os

def save_img(img_arr, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(img_arr, mode='RGB').save(os.path.join(output_dir, file_name))