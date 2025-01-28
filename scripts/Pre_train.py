
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
##################### Network Parameters ###################################
size_input = 1000
patch_size = 48
learning_rate = 0.0001
iterations = int(1e5)
batch_size = 10
save_model_path = "./model3/"
model_name = 'model-epoch'
sample_dir = './sample3'
train_csv_file = './metrics_try_train.csv'
validation_csv_file = './metrics_try_validation.csv'

# Cache file paths
train_cache_file = "cached_train_data.npy"
train_label_cache_file = "cached_train_labels.npy"
val_cache_file = "cached_val_data.npy"
val_label_cache_file = "cached_val_labels.npy"
############################################################################

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

    # Scan the train directory for subdirectories and files
    train_inputs_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Train/Inputs"
    train_labels_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Train/Labels"

    for folder in os.listdir(train_inputs_dir):
        input_dir = os.path.join(train_inputs_dir, folder)
        if not os.path.isdir(input_dir):
            continue

        input_files = os.listdir(input_dir)
        for file in input_files:
            input_path = join(input_dir, file)
            print(input_path)
            in_img = img.imread(input_path)
            in_img = cv2.resize(in_img, (size_input, size_input))

            gt_file = join(train_labels_dir, f"{folder}.jpg")
            if not os.path.exists(gt_file):
                continue
            label_img = img.imread(gt_file)
            label_img = cv2.resize(label_img, (size_input, size_input))

            label_img_stack.append(label_img)
            in_img_stack.append(in_img)

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

    # Scan the validation directory for input and label files
    validation_inputs_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Validation/Inputs"
    validation_labels_dir = "D:/sharef/DSP Code Final Version/Sharef_Code/Processed_Data/Validation/Labels"

    for file in os.listdir(validation_inputs_dir):
        input_path = os.path.join(validation_inputs_dir, file)
        if not os.path.isfile(input_path):
            continue

        print(f"Loading input: {input_path}")
        val_img = img.imread(input_path)
        val_img = cv2.resize(val_img, (size_input, size_input))

        gt_file = os.path.join(validation_labels_dir, file)
        if not os.path.exists(gt_file):
            print(f"Label not found for input: {input_path}")
            continue
        val_label = img.imread(gt_file)
        val_label = cv2.resize(val_label, (size_input, size_input))

        val_label_stack.append(val_label)
        val_img_stack.append(val_img)

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
        in_img = in_img_stack[r_idx]
        h, w, _ = in_img.shape
        x = random.randint(0, h - patch_size)
        y = random.randint(0, w - patch_size)
        if np.max(in_img) > 1:
            in_img = np.array(in_img, dtype="float32") / 255.0
        label_img = label_img_stack[r_idx]
        if np.max(label_img) > 1:
            label_img = np.array(label_img, dtype="float32") / 255.0
        Data[j, :, :, :] = in_img[x:x+patch_size, y:y+patch_size, :]
        Label[j, :, :, :] = label_img[x:x+patch_size, y:y+patch_size, :]

    return Data, Label

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

def concat(layers):
    return tf.concat(layers, axis=3)

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

def validate_model(sess, images, labels, loss, recon_loss, recon_mutal_loss, equal_R_loss, validation_data, validation_labels):
    val_loss, val_recon_loss, val_recon_mutal_loss, val_equal_R_loss = 0, 0, 0, 0
    val_steps = (len(validation_data) + batch_size - 1) // batch_size

    if val_steps == 0:
        print("No validation data available.")
        return 0, 0, 0, 0

    for i in range(val_steps):
        val_batch = validation_data[i * batch_size:(i + 1) * batch_size] / 255.0
        val_label_batch = validation_labels[i * batch_size:(i + 1) * batch_size] / 255.0

        if len(val_batch) == 0 or len(val_label_batch) == 0:
            continue

        val_loss_batch, val_recon_loss_batch, val_recon_mutal_batch, val_equal_R_batch = sess.run(
            [loss, recon_loss, recon_mutal_loss, equal_R_loss],
            feed_dict={images: val_batch, labels: val_label_batch}
        )

        val_loss += val_loss_batch / val_steps
        val_recon_loss += val_recon_loss_batch / val_steps
        val_recon_mutal_loss += val_recon_mutal_batch / val_steps
        val_equal_R_loss += val_equal_R_batch / val_steps

    return val_loss, val_recon_loss, val_recon_mutal_loss, val_equal_R_loss

if __name__ == '__main__':
    DecomNet_layer_num = 5

    images = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
    labels = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

    R_low, I_low = inference(images, layer_num=DecomNet_layer_num)
    R_high, I_high = inference(labels, layer_num=DecomNet_layer_num)

    I_low_3 = concat([I_low, I_low, I_low])
    I_high_3 = concat([I_high, I_high, I_high])

    recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - images))
    recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - labels))
    recon_loss = recon_loss_low + recon_loss_high

    recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - images))
    recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - labels))
    recon_mutal_loss = recon_loss_mutal_low + recon_loss_mutal_high

    equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
    loss = recon_loss + 0.001 * recon_mutal_loss + 0.01 * equal_R_loss

    tf.summary.scalar("Recon_loss", recon_loss)
    tf.summary.scalar("Recon_mutal_loss", recon_mutal_loss)
    tf.summary.scalar("equal_R_loss", equal_R_loss)
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()

    lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    g_optim = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=5)

    # Enable GPU memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    session = InteractiveSession(config=config)

    train_data, train_labels = read_data()
    validation_data, validation_labels = read_validation_data()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer_1 = tf.summary.FileWriter('train3', sess.graph)

        if tf.train.get_checkpoint_state(save_model_path):
            ckpt = tf.train.latest_checkpoint(save_model_path)
            saver.restore(sess, ckpt)
            print(f"Loaded model checkpoint: {ckpt}")
            ckpt_match = re.search(r'-(\d+)$', ckpt)
            start_point = int(ckpt_match.group(1)) if ckpt_match else 0
        else:
            start_point = 0
            print("Starting training from scratch.")

        if not os.path.exists(train_csv_file):
            with open(train_csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Active_Loss", "Recon_Loss", "Recon_Mutal_Loss", "Eq_R_Loss"])

        if not os.path.exists(validation_csv_file):
            with open(validation_csv_file, "w", newline="") as val_f:
                val_writer = csv.writer(val_f)
                val_writer.writerow(["Step", "Validation_Loss", "Validation_Recon_Loss", "Validation_Mutal_Loss", "Validation_Eq_R_Loss"])

        for step in range(iterations):
            train_batch, train_label_batch = batch_read(train_data, train_labels)
            _, train_loss, recon_val, recon_mutal_val, eq_R_val, summary_str = sess.run(
                [g_optim, loss, recon_loss, recon_mutal_loss, equal_R_loss, merged],
                feed_dict={images: train_batch, labels: train_label_batch, lr: learning_rate}
            )
            writer_1.add_summary(summary_str, step)

            print(f"Step {step + 1}/{iterations}, Training_Loss: {train_loss:.4f}, Recon_Loss: {recon_val:.4f}, "
                  f"Recon_Mutal_Loss: {recon_mutal_val:.4f}, Eq_R_Loss: {eq_R_val:.4f}")

            with open(train_csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step + 1, train_loss, recon_val, recon_mutal_val, eq_R_val])

            if np.mod(step + 1, 100) == 0 and step != 0:
                val_loss, val_recon_loss, val_recon_mutal_loss, val_eq_R_loss = validate_model(
                    sess, images, labels, loss, recon_loss, recon_mutal_loss, equal_R_loss, validation_data, validation_labels
                )

                train_sample = train_read(train_data)
                train_sample_R, train_sample_I = sess.run(
                    [R_low, I_low_3], feed_dict={images: train_sample}
                )
                # Save combination of reflectance and illumination for training
                save_images(os.path.join(sample_dir, f'train_eval_{step}.png'), train_sample_R, train_sample_I)

                # Composite image saving for input, reflectance, and illumination
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

                # Composite image saving for input, reflectance, and illumination for validation
                save_images(os.path.join(sample_dir, f'val_composite_{step}.png'), val_sample[0], val_sample_R[0], val_sample_I[0])

                # Save reflectance and illumination for validation
                save_individual(os.path.join(reflectance_dir, f'val_reflectance_{step}.png'), val_sample_R)
                save_individual(os.path.join(illumination_dir, f'val_illumination_{step}.png'), val_sample_I)

                print(f"Validation - Loss: {val_loss:.4f}, Recon_Loss: {val_recon_loss:.4f}, "
                      f"Recon_Mutal_Loss: {val_recon_mutal_loss:.4f}, Eq_R_Loss: {val_eq_R_loss:.4f}")
                      
                with open(validation_csv_file, "a", newline="") as val_f:
                    val_writer = csv.writer(val_f)
                    val_writer.writerow([step + 1, val_loss, val_recon_loss, val_recon_mutal_loss, val_eq_R_loss])

                save_path = os.path.join(save_model_path, model_name)
                saver.save(sess, save_path, global_step=step + 1)

    print("Training completed.")

