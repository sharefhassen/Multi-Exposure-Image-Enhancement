
import os
import re
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import csv
import cv2
import matplotlib.image as img
from os.path import join
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##################### Network Parameters ###################################
size_input = 1000
patch_size = 48
learning_rate = 0.0001
iterations = int(1e5)
batch_size = 10
save_model_path = "./model3/"
model_name = 'model-epoch'
sample_dir = './sample4'
csv_file = './metrics_Decom_train.csv'
validation_csv_file = './metrics_Decom_Validation.csv'
# Cache file paths
train_cache_file = "cached_train_data.npy"
train_label_cache_file = "cached_train_labels.npy"
val_cache_file = "cached_val_data.npy"
val_label_cache_file = "cached_val_labels.npy"
############################################################################
tf.reset_default_graph()

# Ablation mode - Select 'loss', 'recon_loss', or 'smoothness_loss'
LOSS_MODE = 'loss'

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path )

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

def save_images(filepath, input_img, reflectance, illumination=None):
    """Save composite images and optionally reflectance/illumination separately."""
    input_img = np.squeeze(input_img)
    reflectance = np.squeeze(reflectance)

    if illumination is not None and illumination.size > 0:
        illumination = np.squeeze(illumination)
        combined = np.concatenate([input_img, reflectance, illumination], axis=1)
    else:
        combined = np.concatenate([input_img, reflectance], axis=1)

    im = Image.fromarray(np.clip(combined * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def save_individual(filepath, image):
    """Save individual reflectance or illumination image."""
    image = np.squeeze(image)
    im = Image.fromarray(np.clip(image * 255.0, 0, 255.0).astype('uint8'))
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

def inference(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name=f'activated_layer_{idx}')
        conv = tf.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')

    R = tf.sigmoid(conv[:, :, :, 0:3])
    L = tf.sigmoid(conv[:, :, :, 3:4])

    return R, L

def validate_model(sess, images, labels, active_loss, recon_loss, Ismooth_loss, validation_data, validation_labels):
    val_loss, val_recon_loss, val_smoothness_loss = 0, 0, 0
    val_steps = len(validation_data) // batch_size

    for i in range(val_steps):
        val_batch = validation_data[i * batch_size:(i + 1) * batch_size] / 255.0
        val_label_batch = validation_labels[i * batch_size:(i + 1) * batch_size] / 255.0

        val_loss_batch, val_recon_loss_batch, val_smoothness_loss_batch = sess.run(
            [active_loss, recon_loss, Ismooth_loss],
            feed_dict={images: val_batch, labels: val_label_batch}
        )

        val_loss += val_loss_batch
        val_recon_loss += val_recon_loss_batch
        val_smoothness_loss += val_smoothness_loss_batch

    val_loss /= val_steps
    val_recon_loss /= val_steps
    val_smoothness_loss /= val_steps

    return val_loss, val_recon_loss, val_smoothness_loss

if __name__ == '__main__':
    DecomNet_layer_num = 5

    images = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
    labels = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

    R_low, I_low = inference(images, layer_num=DecomNet_layer_num)
    R_high, I_high = inference(labels, layer_num=DecomNet_layer_num)

    I_low_3 = concat([I_low, I_low, I_low])
    I_high_3 = concat([I_high, I_high, I_high])
    # Reconstruction loss
    recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - images))
    recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - labels))
    recon_loss = recon_loss_low + recon_loss_high
    # Smoothness loss
    Ismooth_loss_low = smooth(I_low, R_low)
    Ismooth_loss_high = smooth(I_high, R_high)
    Ismooth_loss = Ismooth_loss_low + Ismooth_loss_high
    # Mutual Reconstruction loss
    recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - images))
    recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - labels))
    recon_mutal_loss = recon_loss_mutal_low + recon_loss_mutal_high
    # Invariable reflectance loss
    equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
    # Ablation Studies
    if LOSS_MODE == 'loss':
        active_loss = recon_loss + 0.1 * Ismooth_loss
    elif LOSS_MODE == 'recon_loss':
        active_loss = recon_loss
    elif LOSS_MODE == 'smoothness_loss':
        active_loss = 0.1 * Ismooth_loss
    else:
        active_loss = recon_loss + 0.1 * Ismooth_loss

    active_loss += 0.001 * (recon_mutal_loss) + 0.01 * equal_R_loss

    tf.summary.scalar("Recon_loss", recon_loss)
    tf.summary.scalar("Smoothness_loss", Ismooth_loss)
    tf.summary.scalar("loss", active_loss)
    merged = tf.summary.merge_all()

    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    g_optim = tf.train.AdamOptimizer(lr).minimize(active_loss)

    saver = tf.train.Saver(max_to_keep=5)

    # Configure GPU memory growth and threading
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4

    train_data, train_labels = read_data()
    validation_data, validation_labels = read_validation_data()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer_1 = tf.summary.FileWriter('train4', sess.graph)

        if tf.train.get_checkpoint_state(save_model_path):
            ckpt = tf.train.latest_checkpoint(save_model_path)
            saver.restore(sess, ckpt)
            print(f"Loaded model checkpoint: {ckpt}")
            ckpt_match = re.search(r'-(\d+)$', ckpt)
            if ckpt_match:
                start_point = int(ckpt_match.group(1))
            else:
                start_point = 0
        else:
            start_point = 0
            print("Starting training from scratch.")

        if not os.path.exists(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Active_Loss", "Recon_Loss", "Smoothness_Loss"])

        if not os.path.exists(validation_csv_file):
            with open(validation_csv_file, "w", newline="") as val_f:
                val_writer = csv.writer(val_f)
                val_writer.writerow(["Step", "Validation_Loss", "Validation_Recon_Loss", "Validation_Smoothness_Loss"])

        for step in range(iterations):
            train_batch, train_label_batch = batch_read(train_data, train_labels)
            _, train_loss, recon_loss_val, smoothness_loss_val, summary_str = sess.run(
                [g_optim, active_loss, recon_loss, Ismooth_loss, merged],
                feed_dict={images: train_batch, labels: train_label_batch, lr: learning_rate}
            )
            writer_1.add_summary(summary_str, step)

            print(f"Step {step + 1}/{iterations}, Active_Loss: {train_loss:.4f}, Recon_Loss: {recon_loss_val:.4f}, Smoothness_Loss: {smoothness_loss_val:.4f}")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step + 1, train_loss, recon_loss_val, smoothness_loss_val])

            if np.mod(step + 1, 100) == 0 and step != 0:
                train_sample = train_read(train_data)
                train_sample_R, train_sample_I = sess.run(
                    [R_low, I_low_3], feed_dict={images: train_sample}
                )
                # Save combination of reflectance and illumination for training
                save_images(os.path.join(sample_dir, f'train_eval_{step}.png'), train_sample_R, train_sample_I)

                # Composite image saving of input, reflectance, and illumination for training
                save_images(os.path.join(sample_dir, f'train_composite_{step}.png'), train_sample[0], train_sample_R[0], train_sample_I[0])

                # Save reflectance and illumination separately
                reflectance_dir = os.path.join(sample_dir, 'Reflectance')
                illumination_dir = os.path.join(sample_dir, 'Illumination')
                
                os.makedirs(reflectance_dir, exist_ok=True)
                os.makedirs(illumination_dir, exist_ok=True)
                
                # Save reflectance and illumination for training
                save_individual(os.path.join(reflectance_dir, f'train_reflectance_{step}.png'), train_sample_R)
                save_individual(os.path.join(illumination_dir, f'train_illumination_{step}.png'), train_sample_I)

                val_sample = valid_read(validation_data)
                val_sample_R, val_sample_I = sess.run(
                    [R_low, I_low_3], feed_dict={images: val_sample}
                )
                # Save combination of reflectance and illumination for validation
                save_images(os.path.join(sample_dir, f'val_eval_{step}.png'), val_sample_R, val_sample_I)

                # Composite image saving of input, reflectance, and illumination for validation
                save_images(os.path.join(sample_dir, f'val_composite_{step}.png'), val_sample[0], val_sample_R[0], val_sample_I[0])

                # Save reflectance and illumination for validation
                save_individual(os.path.join(reflectance_dir, f'val_reflectance_{step}.png'), val_sample_R)
                save_individual(os.path.join(illumination_dir, f'val_illumination_{step}.png'), val_sample_I)

                val_loss, val_recon_loss, val_smoothness_loss = validate_model(
                    sess, images, labels, active_loss, recon_loss, Ismooth_loss, validation_data, validation_labels
                )
                print(f"Validation - Loss: {val_loss:.4f}, Recon_Loss: {val_recon_loss:.4f}, Smoothness_Loss: {val_smoothness_loss:.4f}")

                with open(validation_csv_file, "a", newline="") as val_f:
                    val_writer = csv.writer(val_f)
                    val_writer.writerow([step + 1, val_loss, val_recon_loss, val_smoothness_loss])

                # Save model checkpoint
                save_path = os.path.join(save_model_path, model_name)
                saver.save(sess, save_path, global_step=step + 1)

        print("Training completed.")   

