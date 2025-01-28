
import os
import cv2
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import csv
from os.path import join

# Paths are Indoor\OverExposed or UnderExposed and Outdoor\OverExposed or UnderExposed
base_dir = r"F:\Sharef_Code\Processed_Data\Final_metrics_results\Indoor\OverExposed"  
ground_truth_dir = join(base_dir, "Label")
methods = ["CVC", "DHECI", "LIME", "NPEA", "OURS"]
csv_output = join(base_dir, "Metrics_Indoor_OverExposed_Comparison.csv")

lpips_metric = lpips.LPIPS(net='alex')

def compute_metrics(enhanced_img, gt_img):
    """Compute PSNR, SSIM, and LPIPS."""
    psnr_value = psnr(enhanced_img, gt_img, data_range=1.0)
    ssim_value = ssim(enhanced_img, gt_img, channel_axis=-1, data_range=1.0)
    lpips_value = lpips_metric(
        lpips.im2tensor(enhanced_img * 255),
        lpips.im2tensor(gt_img * 255)
    ).item()
    return psnr_value, ssim_value, lpips_value

if __name__ == "__main__":
    results = []

    for method in methods:
        method_dir = join(base_dir, method)
        if not os.path.exists(method_dir):
            print(f"Method directory not found: {method}")
            continue

        print(f"Processing {method}...")
        psnr_total, ssim_total, lpips_total = 0, 0, 0
        count = 0

        for file in os.listdir(method_dir):
            base_name = file.split('_')[0]  # Extract the label base name
            gt_file = f"{base_name}.jpg"
            gt_path = join(ground_truth_dir, gt_file)

            enhanced_path = join(method_dir, file)
            enhanced_img = cv2.imread(enhanced_path)
            gt_img = cv2.imread(gt_path)

            if enhanced_img is None or gt_img is None:
                print(f"Warning: Unable to read {file} or corresponding label {gt_file}")
                continue

            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Ensure dimensions match by resizing ground truth to match enhanced image
            if enhanced_img.shape != gt_img.shape:
                gt_img = cv2.resize(gt_img, (enhanced_img.shape[1], enhanced_img.shape[0]))

            psnr_value, ssim_value, lpips_value = compute_metrics(enhanced_img, gt_img)

            psnr_total += psnr_value
            ssim_total += ssim_value
            lpips_total += lpips_value
            count += 1

        if count > 0:
            avg_psnr = psnr_total / count
            avg_ssim = ssim_total / count
            avg_lpips = lpips_total / count
            results.append([method, avg_psnr, avg_ssim, avg_lpips])

    with open(csv_output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Avg PSNR", "Avg SSIM", "Avg LPIPS"])
        writer.writerows(results)

    print(f"Metrics comparison saved to {csv_output}")
