clc;
close all;
clear;

rng(5, 'twister'); % seeding the random numbers
x_length = 1024; % lenght of vector
x_spikes = 50; % sparsity of vector
meusurements = 190; % number of measurements
spike_len = 5; % variance of spikes
noiseLevel = 0.2; % noise Level


% generating matrix with given sparsity
pos = randperm(x_length);
x = zeros(x_length,1);
x(pos(1:x_spikes)) = spike_len * randn(x_spikes, 1);

% Generating a sensing matrix
A = randn(meusurements, x_length);
A = orth(A')';

% Generating a noise vector
noise = noiseLevel * randn(meusurements, 1);

% creating the meausurement vector
y = A * x + noise;

% calculating the first 3 reconstructions
x_restored = weighted_l1(y,A,0, 0.1);
x_weighted_1 = weighted_l1(y, A, 1, 0.1);
x_weighted_2 = weighted_l1(y, A, 2, 0.1);

% generating a image
fig1 = figure;

tiledlayout(2,2);

nexttile;
plot(x);
title("original signal");
xlabel("(a)");
xlim([0 x_length]);
ylim([ -spike_len spike_len]);

nexttile;
plot(x_restored);
title("unweighted Reconstruction");
xlabel("(b)");
xlim([0 x_length]);
ylim([-spike_len spike_len]);


nexttile;
plot(x_weighted_1);
title(["Weighted Reconstruction with"," one iteration"]);
xlabel("(c)");
xlim([0 x_length]);
ylim([-spike_len spike_len]);


nexttile;
plot(x_weighted_2);
title(["Weighted Reconstruction with "," two iterations"]);
xlabel("(d)");
xlim([0 x_length]);
ylim([-spike_len spike_len]);


% calculating error for first 6 terms can be changed by changeing limit in
% for
error = [];
for i = 1:6
    x_res = weighted_l1(y, A, i - 1, 0.1);
    error(i) = sqrt(mean((x - x_res).^2));
end

fig2 = figure;
plot(error, "-o");
title("RMSE");
xlabel("number of iterations");
ylabel("error");


% OMP function not needed here
function [ theta  ] = OMP( y,A,K )
    [m, n] = size(A); % getting dimensions 
    theta=zeros(1,n);   % starting answer will be  
    T=[]; % submatrix of A only support set
    r = y; % remaining vector
    i = 0; % iterator
    pos = []; % support set 
    error = 10000; % error convergence variable

    while i < K && error > 1e-9 % till number of iterations or convergence
        product = abs( A' * r ); % abs of A^T r
        [~,pos]=max(product); % max position
        T=[T,A(:,pos)]; % add to submatrix for support set
        i = i + 1; % increase iterator
        pos_a(i) = pos; % add to support set
        newy = pinv(T) * y; % calculate new y
        r = y-T*newy;   % calculate remaining vector
        error = norm(r,2); % error
    end  
    theta(pos_a) = newy; % put to output
end


% weighted_l1 is the weighted l1 minimization that is used in the main code

function theta = weighted_l1(y, A, l, epsilon)
    [m, n] = size(A); % getting dimensions 
    weights = ones(n, 1); % initial weights
    weightsMatrix = diag(weights); % Create a diagnol matrix
    i = 0; % iterator
    error = 1000000; % error or change in vector
    old_theta = zeros(n, 1); % to find difference
    while i < l+1 && error > 1e-6
        inverseWeightsMatrix = inv(weightsMatrix); % inverse of weigghts
        newA = A * inverseWeightsMatrix; % new sensing matrix
        theta = l1_ls(newA, y, 0.01); % solve the l1 minimization problem
%         theta = ISTA(y, newA);
        theta = inverseWeightsMatrix * theta; % Get the original vector
        weightsMatrix = diag(1 ./(abs(theta) + epsilon)); % update the weights
        error = norm(old_theta - theta)
        old_theta = theta;
        i = i +1;
    end
end


% ISTA function can be used instead of l1_ls
function x = ISTA(y, phi)
%     j = zeros(1, 100);
    x = zeros(size(phi'*y)); % answer at the end
    d = eigs(phi' * phi); % calculate eigs for alpha
    alpha = d(1); 
    t = 1 / (2 * alpha); % to be used later
    error = 1000; % to be used when converges
    i = 0; % iterator
    while  error > 0.00001 && i < 100
        a = phi*x;
%         j(i) = sum(abs(a - y).^2) + 10*sum(abs(x(:)));
        x1 = soft(x + (phi'*(y - a))/alpha, t); % soft thresholding 
        error = norm(x1 - x); 
        x = x1;
        i = i +1;
    end
end

function y = soft(x,t)
    y = sign(x).*( max( 0, abs(x)-t ) );
end


