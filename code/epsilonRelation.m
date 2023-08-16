clc;
close all;
clear;

rng(8, 'twister');


errors = [];


for epsilon = [0.001, 0.01, 0.1, 0.5, 1, 2]

x_length = 1024;
x_spikes = 50;
meusurements = 190;
spike_len = 5;


pos = randperm(x_length);
x = zeros(x_length,1);
x(pos(1:x_spikes)) = spike_len * randn(x_spikes, 1);
A = randn(meusurements, x_length);
A = orth(A')';
y = A * x;

errorq = zeros(1, 6);

for l = 0 : 5
    x_restored = weighted_l1(y, A , l, epsilon )';
    errorq(l+1) = sqrt(mean((x_restored' - x) .^2));
end

errors = [errors; errorq];


end

figure;
plot(errors(:,1));
hold on;
plot(errors(:,2));
hold on;
plot(errors(:,3));
hold on;
plot(errors(:,4));
hold on;
plot(errors(:,5));
hold on;
plot(errors(:,6));
legend('0.001', '0.01', '0.1', '0.5', '1', '2');
xlabel("iterations");
ylabel("RMSE");
title("robustness of epsilon");

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


function theta = weighted_l1(y, A, l, epsilon)
    [m, n] = size(A); % getting dimensions 
    weights = ones(n, 1); % initial weights
    weightsMatrix = diag(weights); % Create a diagnol matrix
    if l == -1
        l = 1000;
    end
    error = 100000;
    inverseWeightsMatrix = inv(weightsMatrix); % inverse of weigghts
    newA = A * inverseWeightsMatrix; % new sensing matrix
    theta = l1_ls(newA, y, 0.01); % solve the l1 minimization problem
    newtheta = inverseWeightsMatrix * theta; % Get the original vector
    weightsMatrix = diag(1 ./(abs(newtheta) + epsilon)); % update the weights
    i = 0;
    while i < (l+1) && error > 1e-6
        inverseWeightsMatrix = inv(weightsMatrix); % inverse of weigghts
        newA = A * inverseWeightsMatrix; % new sensing matrix
        theta = l1_ls(newA, y, 0.01); % solve the l1 minimization problem
        newtheta = inverseWeightsMatrix * theta; % Get the original vector
        weightsMatrix = diag(1 ./(abs(newtheta) + epsilon)); % update the weights
        error = norm(newtheta - theta);
        theta = newtheta;
        i = i +1;
    end
    i
    theta = newtheta;
end

function x = ISTA(y, phi)
%     j = zeros(1, 100);
    x = zeros(size(phi'*y));
    d = eigs(phi' * phi);
    alpha = d(1);
    t = 1 / (2 * alpha);
    error = 1000;
    i = 0;
    while  error > 0.00001 && i < 100
        a = phi*x;
%         j(i) = sum(abs(a - y).^2) + 10*sum(abs(x(:)));
        x1 = soft(x + (phi'*(y - a))/alpha, t);
        error = norm(x1 - x);
        x = x1;
        i = i +1;
    end
end

function y = soft(x,t)
    y = sign(x).*( max( 0, abs(x)-t ) );
end


