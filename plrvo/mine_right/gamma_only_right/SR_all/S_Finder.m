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

            G = func_G(C, eta, k, theta);

            added = binom_coeff * (1 - sample_rate)^(alpha + 1 - eta) * sample_rate^eta * G;

            M = M + added;
        end
        M = log(M);

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

end
