clear; clc;
% from meisam
% k = 40;
% theta = 0.00380952380952381;
% C = 3;
% target_epsilon = 0.362749854323983;

% check -- start --
dataset = 'sst-2';

% functions
% M_u = @(k, theta, t) (1 - t .* theta).^(-k);
% func_G = @(C, eta, k, theta) 0.5 * (M_u(k,theta,(eta-1)*C) + M_u(k,theta,-eta*C)) + (1/(2*(2*eta-1))) * (M_u(k,theta,(eta-1)*C) - M_u(k,theta,-eta*C));


% check lmo

% k, theta, lambda, a, b, a_G, a_E, a_U
% a1: a_G; a3: a_E; a4: a_U

%% eps=0.3
k = 1;
theta = 0.5;
lambda = 5;
a = 1;
b = 2;
C = 1;
a_G = 0.1;
a_E = 0.1;
a_U = 0.1;
target_epsilon = 0.3;


%% eps=0.7
k = 1;
theta = 0.5;
lambda = 5;
a = 5;
b = 6;
C = 1;
a_G = 0.1;
a_E = 0.1;
a_U = 0.1;
target_epsilon = 0.7;

%% eps=2
k = 2;
theta = 1;
lambda = 5;
a = 7;
b = 8;
C = 1;
a_G = 0.1;
a_E = 0.1;
a_U = 0.2;
target_epsilon = 2;

%% eps=3
k = 2;
theta = 1;
lambda = 5;
a = 2;
b = 3;
C = 1;
a_G = 0.1;
a_E = 0.1;
a_U = 0.9;
target_epsilon = 3;


% hyperparams
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

SSS_LMO = [k, theta, lambda, a, b, a_G, a_E, a_U];
% 21: e+u;  22: g+u;  23: g+e;
[S_LMO, param_LMO] = S_Finder_LMO(SSS_LMO, C, array_alpha, dataset, sample_size, batch_size, epoch);
% disp(param_LMO);
disp(S_LMO);





function [S, param] = S_Finder_LMO(SSS, C, array_alpha, dataset, sample_size, batch_size, epoch)
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
%% target_delta
target_delta = 1 / (2 * sample_size);
% target_delta = 1e-10;

steps = ceil(epoch / sample_rate);

[SSS_numRows, ~] = size(SSS);

S = [];
for i = 1:SSS_numRows
    k = SSS(i, 1);
    theta = SSS(i, 2);
    lambda = SSS(i, 3);
    a = SSS(i, 4);
    b = SSS(i, 5);
    a_G = SSS(i, 6);
    a_E = SSS(i, 7);
    a_U = SSS(i, 8);
    
    target_epsilon = inf;
    
    for alpha = 4:256
        log_factorials = [0, cumsum(log(1:alpha+1))];

        M = 0;
        for eta = 0:alpha+1
            log_binom_coeff = log_factorials(alpha + 2) - log_factorials(eta + 1) - log_factorials(alpha + 1 - eta + 1);
            binom_coeff = exp(log_binom_coeff);
            
            
            [G, MGF_exist] = func_G_LMO_3_final(C, eta, k, theta, lambda, a, b, a_G, a_E, a_U);

            if MGF_exist == 0
                S = [];
                param = [];
                % return;
                G = inf;
            end
            
            added = binom_coeff * (1 - sample_rate)^(alpha + 1 - eta) * sample_rate^eta * G;
            %% lmo equation using the following;
            % added = G;

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
% param.G_k = S(1);
% param.G_theta = S(2);
% param.PLRV_distortion = distortion;
% param.target_epsilon = target_epsilon; 

end


function [G, MGF_exist] = func_G_LMO_3_final(C, eta, k, theta, lambda, a, b, a_G, a_E, a_U)
    t1 = (eta-1)*C;
    t2 = -eta*C;
    
    [MGF_G1, MGF_exist_G1] = MGF_gamma(a_G * t1, theta, k);
    [MGF_E1, MGF_exist_E1] = MGF_exp(a_E * t1, lambda);
    [MGF_U1, MGF_exist_U1] = MGF_uniform(a_U * t1, a, b);

    if MGF_exist_G1 == 0 || MGF_exist_E1 == 0 || MGF_exist_U1 == 0
        G = NaN;
        MGF_exist = 0;
        return;
    end

    [MGF_G2, MGF_exist_G2] = MGF_gamma(a_G * t2, theta, k);
    [MGF_E2, MGF_exist_E2] = MGF_exp(a_E * t2, lambda);
    [MGF_U2, MGF_exist_U2] = MGF_uniform(a_U * t2, a, b);

    if MGF_exist_G2 == 0 || MGF_exist_E2 == 0 || MGF_exist_U2 == 0
        G = NaN;
        MGF_exist = 0;
        return;
    end

    G1 = MGF_G1 * MGF_E1 * MGF_U1;
    G2 = MGF_G2 * MGF_E2 * MGF_U2;
    G = 0.5 * (G1 + G2) + (1/(2*(2*eta-1))) * (G1 - G2);
    MGF_exist = 1;
end



function [MGF, MGF_exist] = MGF_gamma(t, theta, k)
    % Function to evaluate (1 - theta * t)^(-k) for t < 1/theta
    % Inputs:
    %   t - The value at which the function is evaluated
    %   theta - Parameter (theta > 0)
    %   k - Parameter (k > 0)
    % Outputs:
    %   value - The value of the function at t
    %   valid - Indicator (1 if the function is valid, 0 otherwise)

    % Check if parameters are valid
    if theta <= 0 || k <= 0
        MGF = NaN; % MGF does not exist, return NaN for clarity
        MGF_exist = 0; % Indicate that MGF does not exist
    end

    % Check if t is within the valid range
    if t < 1/theta
        MGF = (1 - theta * t)^(-k); % Calculate the function
        MGF_exist = 1; % Indicate the function is valid
    else
        MGF = NaN; % Return NaN for invalid range
        MGF_exist = 0; % Indicate the function is not valid
    end
end



function [MGF, MGF_exist] = MGF_uniform(t, a, b)
    % Function to calculate the Moment Generating Function (MGF)
    % of a Uniform distribution on [a, b].
    % Inputs:
    %   t - Value at which the MGF is evaluated
    %   a, b - Parameters of the Uniform distribution (a < b)
    % Outputs:
    %   MGF - Value of the MGF at t
    %   MGF_exist - Indicator (1 if MGF exists, 0 otherwise)

    % Check if the input range is valid
    if a >= b
        MGF = NaN;
        MGF_exist = 0;
        return;
    end

    % Calculate the MGF based on the value of t
    if t == 0
        MGF = 1;  % MGF(t=0) is always 1
        MGF_exist = 1;
    else
        MGF = (exp(t*b) - exp(t*a)) / (t * (b - a));
        MGF_exist = 1;
    end
end


function [MGF, MGF_exist] = MGF_exp(t, lambda)
    % Function to calculate the Moment Generating Function (MGF)
    % of an Exponential distribution with rate parameter lambda.
    % Inputs:
    %   t - Value at which the MGF is evaluated
    %   lambda - Rate parameter of the Exponential distribution (lambda > 0)
    % Outputs:
    %   MGF - Value of the MGF at t
    %   MGF_exist - Indicator (1 if MGF exists, 0 otherwise)

    % Check if lambda is valid
    if lambda <= 0
        MGF = NaN; % MGF does not exist, return NaN for clarity
        MGF_exist = 0; % Indicate that MGF does not exist
    end

    % Compute the MGF based on the condition t < lambda
    if t < lambda
        MGF = lambda / (lambda - t); % Valid MGF calculation
        MGF_exist = 1; % MGF exists
    else
        MGF = NaN; % MGF does not exist, return NaN for clarity
        MGF_exist = 0; % Indicate that MGF does not exist
    end
end
