clc;
clear;
close all;

meusurements = 30; % number of measurements out 64 
step = 1; % number of pixels the patch moves
imageSize = 64; % Size of image
start = 128 - 32; % start for the subimage

img = im2double(imread("./barbara256.png")); % Read the image 
img = img(start:imageSize + start, start:imageSize + start); % Get the subImage

[m,n] = size(img);
DCT = kron(dctmtx(8), dctmtx(8)); % Calculate the DCT BASIS
Phi = randn(meusurements,64); % The Meusurement matrix

restored_image = zeros(m, n); % TO store the measuremetnt matrix without weight
restored_image_weighted = zeros(m, n); % REstored image with weight
freq = zeros(m, n); % freq of number of pathces


for i = 1:step:m-7
    for j = 1:step:n-7 % Iterating through patches
        patch = img(i:i+7, j:j+7); % getting a patch 
        patchVector =  reshape(patch, 1, [])'; % vectorizing
        y = Phi * patchVector; % Calculating the meusurement matrix
        restored_patch = weighted_l1(y, Phi * DCT, 0, 0.1); % nuweighted reconstruction
        restored_patch_weighted = weighted_l1(y, Phi* DCT, 4, 0.1); % weighted reconstruction
        
        % adding to the required matirx
        restored_image_weighted(i:i+7, j:j+7) = restored_image_weighted(i:i+7, j:j +7) + reshape(DCT*restored_patch_weighted, 8, 8);
        restored_image(i:i+7, j:j+7) = restored_image(i:i+7, j:j +7) + reshape(DCT*restored_patch, 8, 8);
        freq(i:i+7, j:j+7) =  freq(i:i+7, j:j+7) + 1;
    end
    i
end


% Normalizing
restored_image_weighted = restored_image_weighted ./ freq;
restored_image = restored_image ./ freq;

% Showing the figure
fig =figure;
subplot(1,3,1);
imshow(img)
title("original Image");
xlabel("(a)");


subplot(1,3,2);
imshow(restored_image);
title(["image reconstruction"," without Weight"]);
xlabel("(b)");

subplot(1,3,3);
imshow(restored_image_weighted);
title(["image reconstruction"," with Weighted minimization"]);
xlabel("(c)");






function theta = weighted_l1(y, A, l, epsilon)
    [m, n] = size(A); % getting dimensions 
    weights = ones(n, 1); % initial weights
    weightsMatrix = diag(weights); % Create a diagnol matrix
    
    for i = 1: l + 1
        inverseWeightsMatrix = inv(weightsMatrix); % inverse of weigghts
        newA = A * inverseWeightsMatrix; % new sensing matrix
        theta = l1_ls(newA, y, 0.01, 1e-2, true); % solve the l1 minimization problem
        theta = inverseWeightsMatrix * theta; % Get the original vector
        weightsMatrix = diag(1 ./(abs(theta) + epsilon)); % update the weights
    end
end
