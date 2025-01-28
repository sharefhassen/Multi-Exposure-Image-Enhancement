
import os
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from os.path import join
from PIL import Image
import tr_train as Net

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Paths and Parameters
input_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Test/Inputs/"
output_dir = "./Decom_test_results/"
checkpoint_dir = "./Decom_test_model/"
size_input = 1000  
batch_size = 10   

os.makedirs(output_dir, exist_ok=True)

def concat(layers):
    return tf.concat(layers, axis=3)

def read_test_images(input_dir):
    """Read and preprocess images from the test directory."""
    test_images = []
    file_names = []
    subfolders = []

    subfolders_list = sorted(os.listdir(input_dir))
    folder_count = 0
    for subfolder in subfolders_list:
        subfolder_path = os.path.join(input_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        folder_count += 1
        input_files = sorted(os.listdir(subfolder_path))
        for file in input_files:
            input_file_path = os.path.join(subfolder_path, file)
            print(f"Reading file: {input_file_path}")
            in_img = cv2.imread(input_file_path)
            if in_img is not None:
                in_img = cv2.resize(in_img, (size_input, size_input)).astype(np.float32) / 255.0
                test_images.append(in_img)
                file_names.append(file)
                subfolders.append(subfolder)
            else:
                print(f"Warning: Unable to read {input_file_path}")

    print(f"Total folders loaded: {folder_count}")
    return np.array(test_images), file_names, subfolders

def save_image(filepath, image):
    """Save a single image."""
    image = np.clip(image * 255.0, 0, 255.0).astype(np.uint8)
    Image.fromarray(image).save(filepath)

def save_composite(filepath, input_image, reflectance, illumination):
    """Save composite image with input, reflectance, and illumination side-by-side."""
    input_image = np.clip(input_image * 255.0, 0, 255.0).astype(np.uint8)
    reflectance = np.clip(reflectance * 255.0, 0, 255.0).astype(np.uint8)
    illumination = np.clip(illumination * 255.0, 0, 255.0).astype(np.uint8)

    composite = np.concatenate([input_image, reflectance, illumination], axis=1)
    Image.fromarray(composite).save(filepath)

if __name__ == "__main__":
    test_images, file_names, subfolders = read_test_images(input_dir)

    tensor_input = tf.placeholder(tf.float32, shape=(None, None, None, 3)) 

    # DecomNet inference
    R_out, I_out = Net.inference(tensor_input, layer_num=5)
    I_out = concat([I_out, I_out, I_out])

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4

    with tf.Session(config=config) as sess:
        try:
            print("[*] Attempting to load the latest model checkpoint...")
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print(f"Model successfully restored from: {ckpt}")
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {checkpoint_dir}")
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            exit()

        num_batches = len(test_images) // batch_size + (1 if len(test_images) % batch_size > 0 else 0)

        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = min((batch_index + 1) * batch_size, len(test_images))
            input_batch = test_images[start:end]

            reflectance_batch, illumination_batch = sess.run([R_out, I_out], feed_dict={tensor_input: input_batch})

            for i in range(len(input_batch)):
                reflectance = np.squeeze(reflectance_batch[i])
                illumination = np.squeeze(illumination_batch[i])

                subfolder = subfolders[start + i]
                file_name = file_names[start + i]

                reflectance_folder = join(output_dir, "Reflectance", subfolder)
                illumination_folder = join(output_dir, "Illumination", subfolder)
                composite_folder = join(output_dir, "Composite", subfolder)

                os.makedirs(reflectance_folder, exist_ok=True)
                os.makedirs(illumination_folder, exist_ok=True)
                os.makedirs(composite_folder, exist_ok=True)

                # Save reflectance
                reflectance_path = join(reflectance_folder, f"{os.path.splitext(file_name)[0]}.png")
                save_image(reflectance_path, reflectance)

                # Save illumination
                illumination_path = join(illumination_folder, f"{os.path.splitext(file_name)[0]}.png")
                save_image(illumination_path, illumination)

                # Save composite
                composite_path = join(composite_folder, f"{os.path.splitext(file_name)[0]}.png")
                save_composite(composite_path, input_batch[i], reflectance, illumination)

                print(f"Processed {file_name}: Reflectance, Illumination, and Composite saved.")

    print("Decomposition completed.")

