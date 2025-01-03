clear; clc;
%% from meisam
% k = 40;
% theta = 0.00380952380952381;
% C = 3;
% target_epsilon = 0.362749854323983;

k = 5;
theta = 0.261307;
C = 0.9;
target_epsilon = 0.362749854323983;

%% check -- start --
dataset = 'sst-2';
dataset = 'p100';



%% functions
M_u = @(k, theta, t) (1 - t .* theta).^(-k);
func_G = @(C, eta, k, theta) 0.5 * (M_u(k,theta,(eta-1)*C) + M_u(k,theta,-eta*C)) + (1/(2*(2*eta-1))) * (M_u(k,theta,(eta-1)*C) - M_u(k,theta,-eta*C));

array_alpha = 2:256;

dict = containers.Map();
% sample_size, batch_size, epoch
dict('sst-2') = [67349, 1024, 3];
dict('mnli') = [392703, 1024, 3];
dict('qnli') = [104743, 1024, 3];
dict('qqp') = [363847, 1024, 3];
dict('e2e') = [42061, 1024, 10];
dict('dart') = [93187, 1024, 10];
dict('cifar10') = [50000, 1024, 10]; % cifar100;
dict('mnist') = [60000, 1024, 10]; % fmnist; kmnist;
dict('p100') = [10000, 1024, 10];


params = dict(dataset);
sample_size = params(1);
batch_size = params(2);
epoch = params(3);

fprintf('Dataset: %s, Sample Size: %d, Batch Size: %d, Epoch: %d\n', dataset, sample_size, batch_size, epoch);

SSS = [k, theta];
[S, param] = S_Finder(SSS, C, array_alpha, dataset, sample_size, batch_size, epoch, func_G);
disp(param);
confirmed = boolean((param.target_epsilon - target_epsilon ) <1e-5);
fprintf("Confirmed: %d", confirmed)

function [S, param] = S_Finder(SSS, C, array_alpha, dataset, sample_size, batch_size, epoch, func_G)
% S_FINDER: The main program for searching the PLRV-O noise parameters.
% Inputs:
%   SSS: A matrix where each row contains [k, theta] values in superior search space.
%   C: The clipping threshold (e.g., 0.1).
%   array_alpha: An array of accountant orders (e.g., 2:256).
%   dataset: Name of the dataset.
%   sample_size: Total number of samples.
%   batch_size: Size of each batch.
%   epoch: Number of training epochs.
%   func_G: function of PLRV-O accountant.
% Outputs:
%   S: Search results (k, theta, distortion).
%   param: A structure containing hyperparameters.


sample_rate = batch_size / sample_size;
target_delta = 1 / (2 * sample_size);
steps = ceil(epoch / sample_rate);

[SSS_numRows, ~] = size(SSS);

S = [];
for i = 1:SSS_numRows
    k = SSS(i, 1);
    theta = SSS(i, 2);
    
    target_epsilon = inf;
    
    for alpha = array_alpha
        log_factorials = [0, cumsum(log(1:alpha+1))];

        M = 0;
        for eta = 0:alpha+1
            log_binom_coeff = log_factorials(alpha + 2) - log_factorials(eta + 1) - log_factorials(alpha + 1 - eta + 1);
            binom_coeff = exp(log_binom_coeff);
            
            if 1 - (eta-1)*C * theta <=0 || 1 - (-eta*C)*theta <=0
                G = inf;
            else
                G = func_G(C, eta, k, theta);
            end

            added = binom_coeff * (1 - sample_rate)^(alpha + 1 - eta) * sample_rate^eta * G;

            M = M + added;
        end
        M = log(M)/alpha;

        if isnan(M) || isinf(M) || M < 0
            continue;
        else
            M = steps * M;
        end

        epsilon = M + log(alpha / (1 + alpha)) - ((log(target_delta) + log(1 + alpha)) / alpha);
        if epsilon < target_epsilon
            target_epsilon = epsilon;
        end
    end
    
    if target_epsilon < 10
        distortion = 1 / (k - 1) * theta;
        S = [S; k, theta, distortion, target_epsilon];
    end
    
end


param.C = C;
param.array_alpha = int32(array_alpha(end));
param.sample_size = int32(sample_size);
param.target_delta = target_delta;
param.batch_size = int32(batch_size);
param.sample_rate = sample_rate;
param.epoch = int32(epoch);
param.steps = int32(steps);
param.dataset = dataset;
param.G_k = S(1);
param.G_theta = S(2);
param.PLRV_distortion = distortion;
param.target_epsilon = target_epsilon; 

end
