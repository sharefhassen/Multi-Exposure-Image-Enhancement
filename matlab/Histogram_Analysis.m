clc;
clear;

% CVC
img_cvc = imread("CVC\5.jpg");
img_cvc = rgb2gray(img_cvc);
imhist(img_cvc)

% DHECI
img_dheci = imread("dheci\5.jpg");
img_dheci = rgb2gray(img_dheci);
imhist(img_dheci)

% LIME
img_lime = imread("LIME\5.jpg");
img_lime = rgb2gray(img_lime);
imhist(img_lime)

% NPEA
img_npea = imread("NPEA\5.jpg");
img_npea = rgb2gray(img_npea);
imhist(img_npea)

% Ours
img_ours = imread("my method\5_train9.jpg");
img_ours = rgb2gray(img_ours);
imhist(img_ours)

% Input
img_input = imread("Input_Image_Histo_1.jpg");
img_input = rgb2gray(img_input);
imhist(img_input)