clc;
clear;

% Paths to enhanced images and the reference image
enhancedImages = {
    "CVC\79_5.jpg", 
    "dheci\79_5.jpg", 
    "LIME\79_5.jpg", 
    "NPEA\79_5.jpg", 
    "my method\79_5_train9.jpg"
};
methodNames = {'CVC', 'DHECI', 'LIME', 'NPEA', 'Proposed Method'};
referenceImagePath = "79_Label.JPG";

% Define the common dimension
commonHeight = 2189;
commonWidth = 1460;

% Read the reference image and resize it
referenceImage = imread(referenceImagePath);
referenceImageResized = imresize(referenceImage, [commonHeight, commonWidth]);
referenceImageGray = rgb2gray(referenceImageResized);

% Define fixed diagonal line coordinates 
x = linspace(1, commonWidth, 2000);  % Fixed length for intensity profiles
y = linspace(1, commonHeight, 2000);

% Process each enhanced image
for i = 1:length(enhancedImages)
    % Read enhanced image
    enhancedImage = imread(enhancedImages{i});

    % Resize to the common dimension
    enhancedImageResized = imresize(enhancedImage, [commonHeight, commonWidth]);

    % Convert to grayscale
    enhancedImageGray = rgb2gray(enhancedImageResized);

    % Extract intensity profiles
    enhancedProfile = improfile(enhancedImageGray, x, y, 2000);
    referenceProfile = improfile(referenceImageGray, x, y, 2000);

    % Create a figure for the current method
    figure;
    plot(enhancedProfile, 'r', 'LineWidth', 1.5); % Enhanced image profile
    hold on;
    plot(referenceProfile, 'b--', 'LineWidth', 1.5); % Reference image profile
    xlabel('Distance (pixels)');
    ylabel('Intensity');
    legend(methodNames{i}, 'Reference');
    grid on;
    hold off;
end
