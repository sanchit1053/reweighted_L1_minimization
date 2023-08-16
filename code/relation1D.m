clc;
close all;
clear;

rng(8, 'twister'); % seeding the random numbers
x_length = 1024; % lenght of vector
x_spikes = 50; % sparsity of vector
meusurements = 190; % number of measurements
spike_len = 5; % variance of spikes

% generating matrix with given sparsity
pos = randperm(x_length);
x = zeros(x_length,1);
x(pos(1:x_spikes)) = spike_len * randn(x_spikes, 1);

% Generating a sensing matrix
A = randn(meusurements, x_length);
A = orth(A')';


% creating the meausurement vector
y = A * x;

% calculating the first 4 reconstructions
x_restored = l1_ls(A,y,0.01)';
x_weighted_1 = weighted_OMP(y, A, 1000, 1, 0.1);
x_weighted_2 = weighted_OMP(y, A, 1000, 2, 0.1);
x_weighted_3 = weighted_OMP(y, A, 1000, 3, 0.1);


% generating the image
tiledlayout(2,2);


nexttile;
scatter(x, x_restored,20);
title("correlation of original with unweighted");
xlabel(["x_0" , "(a)"]);
ylabel("x^0");
xlim([-spike_len spike_len]);
ylim([-spike_len spike_len]);
% 
nexttile;
scatter(x, x_weighted_1,20);
title(["correlation of original with "," one iteration weighted"]);
xlabel(["x_0", "(b)"]);
ylabel("x^1");
xlim([-spike_len spike_len]);
ylim([-spike_len spike_len]);
% 

nexttile;
scatter(x, x_weighted_2,20);
title(["correlation of original with ","two iteration weighted"]);
xlabel(["x_0","(c)"]);
ylabel("x^1");
xlim([-spike_len spike_len]);
ylim([-spike_len spike_len]);

nexttile;
scatter(x, x_weighted_3,20);
title(["correlation of original with"," three iteration weighted"]);
xlabel(["x_0","(d)"]);
ylabel("x^3");
xlim([-spike_len spike_len]);
ylim([-spike_len spike_len]);


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


function theta = weighted_OMP(y, A, K, l, epsilon)
    [m, n] = size(A); % getting dimensions 
    weights = ones(n, 1); % initial weights
    weightsMatrix = diag(weights); % Create a diagnol matrix
    
    for i = 1: l + 1
        inverseWeightsMatrix = inv(weightsMatrix); % inverse of weigghts
        newA = A * inverseWeightsMatrix; % new sensing matrix
        theta = l1_ls(newA, y, 0.01); % solve the l1 minimization problem
        theta = inverseWeightsMatrix * theta; % Get the original vector
        weightsMatrix = diag(1 ./(abs(theta) + epsilon)); % update the weights
    end
end