
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import cv2
from os.path import join
from Enhance_train import I_inference, R_inference, concat
import Decom_train as Net
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Paths and Parameters
input_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Test/Inputs/"
label_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Test/Labels/"
output_dir = "./Enhance_test_results/"
checkpoint_dir = "./Enhance_test_model/"
csv_file = "./Enhance_test_metrics.csv"
batch_size = 10  
size_input = 1000

os.makedirs(output_dir, exist_ok=True)

def read_test_data(input_dir, label_dir):
    """Load and preprocess input images and match them with corresponding label images."""
    input_images = []
    label_images = []
    file_names = []
    subfolders = []

    subfolders_list = sorted(os.listdir(input_dir))
    for subfolder in subfolders_list:
        subfolder_path = join(input_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        label_file = join(label_dir, f"{subfolder}.jpg")
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found for subfolder {subfolder}")
            continue

        print(f"Reading inputs from subfolder: {subfolder}, label: {label_file}")
        label_img = cv2.imread(label_file)

        if label_img is None:
            print(f"Warning: Unable to read label for subfolder {subfolder}")
            continue

        label_img = cv2.resize(label_img, (size_input, size_input)).astype(np.float32) / 255.0

        input_files = sorted(os.listdir(subfolder_path))
        for file in input_files:
            input_file_path = join(subfolder_path, file)
            input_img = cv2.imread(input_file_path)

            if input_img is None:
                print(f"Warning: Unable to read input image: {input_file_path}")
                continue

            input_img = cv2.resize(input_img, (size_input, size_input)).astype(np.float32) / 255.0
            input_images.append(input_img)
            label_images.append(label_img)
            file_names.append(file)
            subfolders.append(subfolder)

    print(f"Total Test Inputs: {len(input_images)}, Total Test Labels: {len(label_images)}")
    return np.array(input_images), np.array(label_images), file_names, subfolders


def save_image(filepath, image):
    """Save a single image."""
    image = np.clip(image * 255.0, 0, 255.0).astype(np.uint8)
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_composite(filepath, input_image, enhanced_image, label_image):
    """Save composite image with input, enhanced, and label images side-by-side."""
    input_image = np.clip(input_image * 255.0, 0, 255.0).astype(np.uint8)
    enhanced_image = np.clip(enhanced_image * 255.0, 0, 255.0).astype(np.uint8)
    label_image = np.clip(label_image * 255.0, 0, 255.0).astype(np.uint8)

    composite = np.concatenate([input_image, enhanced_image, label_image], axis=1)
    cv2.imwrite(filepath, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

def save_reflectance_and_illumination(sess, images, R_out, I_out, test_data, file_names, subfolders):
    """Save reflectance and illumination components separately."""
    num_images = len(test_data)

    for i in range(0, num_images, batch_size):
        batch_data = test_data[i:i + batch_size]
        batch_files = file_names[i:i + batch_size]
        batch_subfolders = subfolders[i:i + batch_size]

        reflectance_batch, illumination_batch = sess.run([R_out, I_out], feed_dict={images: batch_data})

        for j, (reflectance, illumination) in enumerate(zip(reflectance_batch, illumination_batch)):
            file_name = batch_files[j]
            subfolder = batch_subfolders[j]

            reflectance_folder = join(output_dir, "Reflectance", subfolder)
            illumination_folder = join(output_dir, "Illumination", subfolder)

            os.makedirs(reflectance_folder, exist_ok=True)
            os.makedirs(illumination_folder, exist_ok=True)
            
            reflectance_path = join(reflectance_folder, f"{os.path.splitext(file_name)[0]}_R.png")
            illumination_path = join(illumination_folder, f"{os.path.splitext(file_name)[0]}_I.png")
            
            save_image(reflectance_path, reflectance)
            save_image(illumination_path, illumination)

            print(f"Processed {file_name}: Reflectance and Illumination saved.")

def calculate_metrics(pred, label):
    """Calculate PSNR, SSIM, and LPIPS."""
    psnr_value = psnr(pred, label, data_range=1.0)
    ssim_value = ssim(pred, label, channel_axis=-1, data_range=1.0)
    lpips_metric = lpips.LPIPS(net='alex')
    lpips_value = lpips_metric(
        lpips.im2tensor(pred * 255),
        lpips.im2tensor(label * 255)
    ).item()
    return psnr_value, ssim_value, lpips_value

def batch_processing(sess, images, output, test_data, test_labels, file_names, subfolders):
    """Process test data in batches and log metrics to CSV."""
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    num_images = len(test_data)

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "Subfolder", "PSNR", "SSIM", "LPIPS"])

        for i in range(0, num_images, batch_size):
            batch_data = test_data[i:i + batch_size]
            batch_labels = test_labels[i:i + batch_size]
            batch_files = file_names[i:i + batch_size]
            batch_subfolders = subfolders[i:i + batch_size]

            batch_outputs = sess.run(output, feed_dict={images: batch_data})

            for j, pred in enumerate(batch_outputs):
                label = batch_labels[j]
                psnr_value, ssim_value, lpips_value = calculate_metrics(pred, label)

                file_name = batch_files[j]
                subfolder = batch_subfolders[j]

                enhanced_folder = join(output_dir, "Enhanced", subfolder)
                composite_folder = join(output_dir, "Composite", subfolder)

                os.makedirs(enhanced_folder, exist_ok=True)
                os.makedirs(composite_folder, exist_ok=True)

                enhanced_path = join(enhanced_folder, f"{os.path.splitext(file_name)[0]}.png")
                composite_path = join(composite_folder, f"{os.path.splitext(file_name)[0]}.png")
                
                save_image(enhanced_path, pred) 
                save_composite(composite_path, batch_data[j], pred, label)

                writer.writerow([file_name, subfolder, psnr_value, ssim_value, lpips_value])

                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value

            print(f"Processed batch {i // batch_size + 1}")

        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        avg_lpips = total_lpips / num_images
        writer.writerow(["Average", "", avg_psnr, avg_ssim, avg_lpips])

    return avg_psnr, avg_ssim, avg_lpips

if __name__ == "__main__":
    test_data, test_labels, file_names, subfolders = read_test_data(input_dir, label_dir)

    images = tf.placeholder(tf.float32, shape=(None, None, None, 3))

    R_out, I_out = Net.inference(images, 5)
    I_output = I_inference(R_out, I_out)
    R_output = R_inference(R_out, I_out)
    I_delta = concat([I_output, I_output, I_output])
    output = R_output * I_delta

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print(f"Model restored from {ckpt}")
        else:
            print("No checkpoint found.")
            exit()

        save_reflectance_and_illumination(sess, images, R_out, I_out, test_data, file_names, subfolders)
        avg_psnr, avg_ssim, avg_lpips = batch_processing(sess, images, output, test_data, test_labels, file_names, subfolders)
        print(f"Final Average Metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
