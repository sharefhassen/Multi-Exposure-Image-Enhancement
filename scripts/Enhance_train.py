
import os
import re
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2
import matplotlib.image as img
from os.path import join
import csv
import time
from PIL import Image
import Decom_train as Net
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import lpips

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##################### Network Parameters ###################################
size_input = 1000
patch_size = 256
learning_rate = 0.00001
iterations = int(1e5)
batch_size = 10
save_model_path = "./enhance_model/"
model_name = 'enhance-epoch'
sample_dir = './enhance_sample'
csv_file = './metrics_Enhance_train.csv'
validation_csv_file = './metrics_Enhance_validation.csv'
# Cache file paths
train_cache_file = "cached_train_data.npy"
train_label_cache_file = "cached_train_labels.npy"
val_cache_file = "cached_val_data.npy"
val_label_cache_file = "cached_val_labels.npy"
############################################################################
tf.reset_default_graph()

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path )

# Ablation mode - Select 'loss', 'Ismooth_loss_delta', or 'relight_loss'
LOSS_MODE = 'loss'

# Data Reading Functions
def read_data():
    # Check for cached data
    if os.path.exists(train_cache_file) and os.path.exists(train_label_cache_file):
        print("Loading cached training data...")
        in_img_stack = np.load(train_cache_file)
        label_img_stack = np.load(train_label_cache_file)
        print(f"Loaded Training Inputs: {len(in_img_stack)}, Labels: {len(label_img_stack)}")
        return in_img_stack, label_img_stack

    in_img_stack = []
    label_img_stack = []

    input_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Train/Inputs"
    label_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Train/Labels"

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = join(root, file)
            label_file = join(label_dir, os.path.splitext(file)[0] + ".jpg") 
            if not os.path.exists(label_file):
                continue
            print(input_path)
            in_img = img.imread(input_path)
            in_img = cv2.resize(in_img, (size_input, size_input))

            label_img = img.imread(label_file)
            label_img = cv2.resize(label_img, (size_input, size_input))

            in_img_stack.append(in_img)
            label_img_stack.append(label_img)

    print(f"Total Training Inputs: {len(in_img_stack)}")
    print(f"Total Training Labels: {len(label_img_stack)}")

    # Save to cache
    np.save(train_cache_file, np.array(in_img_stack))
    np.save(train_label_cache_file, np.array(label_img_stack))

    return np.array(in_img_stack), np.array(label_img_stack)

def read_validation_data():
    # Check for cached data
    if os.path.exists(val_cache_file) and os.path.exists(val_label_cache_file):
        print("Loading cached validation data...")
        val_img_stack = np.load(val_cache_file)
        val_label_stack = np.load(val_label_cache_file)
        print(f"Loaded Validation Inputs: {len(val_img_stack)}, Labels: {len(val_label_stack)}")
        return val_img_stack, val_label_stack

    val_img_stack = []
    val_label_stack = []

    input_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Validation/Inputs"
    label_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Validation/Labels"

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = join(root, file)
            label_file = join(label_dir, os.path.splitext(file)[0] + ".jpg") 
            if not os.path.exists(label_file):
                continue
            print(input_path)
            val_img = img.imread(input_path)
            val_img = cv2.resize(val_img, (size_input, size_input))

            val_label = img.imread(label_file)
            val_label = cv2.resize(val_label, (size_input, size_input))

            val_img_stack.append(val_img)
            val_label_stack.append(val_label)

    print(f"Total Validation Inputs: {len(val_img_stack)}")
    print(f"Total Validation Labels: {len(val_label_stack)}")

    # Save to cache
    np.save(val_cache_file, np.array(val_img_stack))
    np.save(val_label_cache_file, np.array(val_label_stack))

    return np.array(val_img_stack), np.array(val_label_stack)

def batch_read(in_img_stack, label_img_stack):
    Data = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
    Label = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

    for j in range(batch_size):
        r_idx = random.randint(0, len(in_img_stack) - 1)
        in_ = in_img_stack[r_idx]
        h, w, _ = in_.shape
        x = random.randint(0, h - patch_size)
        y = random.randint(0, w - patch_size)
        if np.max(in_) > 1:
            in_ = np.array(in_, dtype="float32") / 255.0
        label_ = label_img_stack[r_idx]
        if np.max(label_) > 1:
            label_ = np.array(label_, dtype="float32") / 255.0
        Data[j, :, :, :] = in_[x:x+patch_size, y:y+patch_size, :]
        Label[j, :, :, :] = label_[x:x+patch_size, y:y+patch_size, :]

    return Data, Label

def train_read(in_img_stack):
    eval_low_data = []
    idx = random.randint(0,len(in_img_stack)-1)
    eval_low_im = in_img_stack[idx]
    if np.max(eval_low_im) > 1:
        eval_low_im = np.array(eval_low_im, dtype="float32") / 255.0
    eval_low_data.append(eval_low_im)
        
    return eval_low_data

def valid_read(val_img_stack):
    eval_val_data = []
    idx = random.randint(0, len(val_img_stack) - 1)
    eval_val_im = val_img_stack[idx]
    eval_val_im = np.array(eval_val_im, dtype="float32") / 255.0 
    eval_val_data.append(eval_val_im)
    return np.array(eval_val_data)

def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    if result_2 is not None:
        result_2 = np.squeeze(result_2)
        cat_image = np.concatenate([result_1, result_2], axis=1)
    else:
        cat_image = result_1

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def concat(layers):
    return tf.concat(layers, axis=3)

def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])

    kernel = smooth_kernel_x if direction == "x" else smooth_kernel_y
    return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

def ave_gradient(input_tensor, direction):
    return tf.layers.average_pooling2d(gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

def smooth(input_I, input_R):
    input_R = tf.image.rgb_to_grayscale(input_R)
    return tf.reduce_mean(gradient(input_I, "x") * tf.exp(-10 * ave_gradient(input_R, "x")) + gradient(input_I, "y") * tf.exp(-10 * ave_gradient(input_R, "y")))

def I_inference(input_R, input_L, channel=16, kernel_size=3):
    input_im = concat([input_R, input_L])
    
    conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        
    up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
    deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
    up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
    deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
    up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
    deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        
    deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
    feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
    output = tf.layers.conv2d(feature_gather, 1, 3, padding='same', activation=None)
    return output

def R_inference(input_R, input_L, channel=16, kernel_size=3):
    input_im = concat([input_R, input_L])
    
    conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    output = tf.layers.conv2d(conv6, 3, 3, padding='same', activation=None)
    return output

def new_net(input, channel=16, kernel_size=3):
    conv0 = tf.layers.conv2d(input, channel, kernel_size, padding='same', activation=None)
    conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, channel, kernel_size, strides=1, padding='same', activation=tf.nn.relu)
    output = tf.layers.conv2d(conv6, 3, 3, padding='same', activation=None);    
    return output

def validate_model(sess, images, labels, output, loss, relight_loss, Ismooth_loss_delta, validation_data, validation_labels):
    val_psnr, val_ssim, val_loss, val_relight_loss, val_smoothness_loss = 0, 0, 0, 0, 0
    num_samples = len(validation_data)
    num_batches = (num_samples + batch_size - 1) // batch_size  

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        val_batch = np.array(validation_data[start_idx:end_idx])
        val_label_batch = np.array(validation_labels[start_idx:end_idx])

        val_batch = val_batch.astype(np.float32) / 255.0 if val_batch.max() > 1 else val_batch.astype(np.float32)
        val_label_batch = val_label_batch.astype(np.float32) / 255.0 if val_label_batch.max() > 1 else val_label_batch.astype(np.float32)

        val_output, val_loss_batch, val_relight_loss_batch, val_smoothness_loss_batch = sess.run(
            [output, loss, relight_loss, Ismooth_loss_delta],
            feed_dict={images: val_batch, labels: val_label_batch}
        )

        # Compute metrics for each image in the batch
        for i in range(len(val_batch)):
            val_psnr += psnr(val_label_batch[i], val_output[i], data_range=1.0)
            val_ssim += ssim(val_label_batch[i], val_output[i], channel_axis=-1)

        # Accumulate batch losses
        val_loss += val_loss_batch
        val_relight_loss += val_relight_loss_batch
        val_smoothness_loss += val_smoothness_loss_batch

    # Average loss calculations
    num_total_images = num_batches * batch_size
    return (
        val_loss / num_batches,  
        val_relight_loss / num_batches,  
        val_smoothness_loss / num_batches, 
        val_psnr / num_samples, 
        val_ssim / num_samples, 
    )


if __name__ == '__main__':
    images = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images')
    labels = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='labels')

    R_out, I_out = Net.inference(images, 5)

    I_output = I_inference(R_out, I_out)
    R_output = R_inference(R_out, I_out)

    I_delta = concat([I_output, I_output, I_output])
    output = R_output * I_delta
    
    # Smoothness loss
    Ismooth_loss_delta = smooth(I_output, R_output)
    # Enhancement loss
    relight_loss = tf.reduce_mean(tf.abs(output - labels))

    # Ablation
    if LOSS_MODE=='loss':
        loss = relight_loss + 0.1 * Ismooth_loss_delta
    elif LOSS_MODE=='Ismooth_loss_delta':
        loss = 0.1 * Ismooth_loss_delta
    elif LOSS_MODE=='relight_loss':
        loss =  relight_loss
    else:
        loss = relight_loss + 0.1 * Ismooth_loss_delta

    tf.summary.scalar("Relight_Loss", relight_loss)
    tf.summary.scalar("Smoothness_Loss", Ismooth_loss_delta)
    tf.summary.scalar("Training_Loss", loss)
    merged = tf.summary.merge_all()

    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    g_optim = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=5)

    # GPU memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4

    train_data, train_labels = read_data()
    validation_data, validation_labels = read_validation_data()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer_1 = tf.summary.FileWriter('enhance_train10', sess.graph)

        if tf.train.get_checkpoint_state(save_model_path):   
            ckpt = tf.train.latest_checkpoint(save_model_path)
            saver.restore(sess, ckpt)
            print(f"Loaded model checkpoint: {ckpt}")
            ckpt_match = re.search(r'-(\d+)$', ckpt)  
            if ckpt_match:
                start_point = int(ckpt_match.group(1))
            else:
                start_point = 0
                print("Checkpoint does not contain a step number. Starting from 0.")
        else:  
            start_point = 0
            print("Starting training from scratch.")

        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Training_Loss", "Relight_Loss", "Smoothness_Loss"])

        if not os.path.exists(validation_csv_file):
            with open(validation_csv_file, "w", newline="") as val_f:
                val_writer = csv.writer(val_f)
                val_writer.writerow(["Step", "Validation_Loss", "Validation_Relight_Loss", "Validation_Smoothness_Loss", "PSNR", "SSIM"])

        for step in range(iterations):
            train_batch, train_label_batch = batch_read(train_data, train_labels)
            _, training_loss_val, relight_val, smooth_val, summary_str = sess.run(
                [g_optim, loss, relight_loss, Ismooth_loss_delta, merged],
                feed_dict={images: train_batch, labels: train_label_batch, lr: learning_rate}
            )
            writer_1.add_summary(summary_str, step)

            print(f"Step {step + 1}/{iterations}, Training_Loss: {training_loss_val:.4f}, Relight_Loss: {relight_val:.4f}, Smoothness_Loss: {smooth_val:.4f}")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step + 1, training_loss_val, relight_val, smooth_val])

            if np.mod(step + 1, 1000) == 0 and step != 0:
                train_sample = np.array(train_read(train_data))
                train_sample = train_sample.astype(np.float32) / 255.0 if train_sample.max() > 1 else train_sample.astype(np.float32)
                R_, I_, out_ = sess.run([R_out, I_delta, output], feed_dict={images: train_sample})
                # Save individual enhanced image for training
                save_images(os.path.join(sample_dir, f'train_eval_{step}.png'), out_[0])

                # Composite saving for training
                save_images(os.path.join(sample_dir, f'train_composite_{step}.png'), train_sample[0], out_[0])

                val_sample = np.array(valid_read(validation_data))
                val_sample = val_sample.astype(np.float32) / 255.0 if val_sample.max() > 1 else val_sample.astype(np.float32)
                val_R, val_I, val_out = sess.run([R_out, I_delta, output], feed_dict={images: val_sample})
                # Save individual enhanced image for validation
                save_images(os.path.join(sample_dir, f'val_eval_{step}.png'), val_out[0])

                # Composite saving for validation
                save_images(os.path.join(sample_dir, f'val_composite_{step}.png'), val_sample[0], val_out[0])

                # Validation metrics
                val_loss, val_relight_loss, val_smooth_loss, val_psnr, val_ssim = validate_model(
                    sess, images, labels, output, loss, relight_loss, Ismooth_loss_delta, validation_data, validation_labels
                )
                print(f"Validation - Loss: {val_loss:.4f}, Relight_Loss: {val_relight_loss:.4f}, Smoothness_Loss: {val_smooth_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")

                with open(validation_csv_file, "a", newline="") as val_f:
                    val_writer = csv.writer(val_f)
                    val_writer.writerow([step + 1, val_loss, val_relight_loss, val_smooth_loss, val_psnr, val_ssim])

                # Save model checkpoint
                save_path = os.path.join(save_model_path, model_name)
                saver.save(sess, save_path, global_step=step + 1)

        print("Training completed.")













