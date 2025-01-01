% 
 % Q=Qn;
% % Step 1: Define filter conditions
% filter_idx = Q(:, 1) < 5;    % Logical index for epsilon < 10 (column 1)
% 
% 
% % Step 2: Combine conditions
% combined_filter = filter_idx; % Logical AND of both conditions
% 
% % Step 3: Apply filter to the matrix
% filtered_Q = Q(combined_filter, :); % Extract rows satisfying both conditions

delta= Q(:,3); 
clip= Q(:,4);
q=Q(:,5);
k = Q(:,6);   
% Extract relevant co                % Column 1: k

theta = Q(:,7);              % Column 2: theta
epsilon = Q(:,1);            % Column 3: epsilon (privacy budget)
distortion = Q(:,2);         % Column 4: distortion

% Step 1: Compute k.theta (x-axis for both plots)
k_theta = k .* theta;

% Step 2: Filter rows where epsilon < 1
% filter_idx = epsilon < 10;    % Logical index for epsilon < 1
% filter_idy = qd<0.002;  
% k_theta_filtered = k_theta(filter_idx);
% epsilon_filtered = epsilon(filter_idx);
% distortion_filtered = distortion(filter_idx);
% theta_filtered = theta(filter_idx);
% log10(distortion_filtered)
% Step 3: Create subplots


% Subplot 1: Privacy Budget (\epsilon)
% Scatter plot with log-scaled theta
figure;
scatter(k_theta, epsilon, 40, log10(distortion), 'filled'); % Scatter plot with log-scaled theta
colormap('hot'); % Set colormap

% Add colorbar for theta
cb1 = colorbar;

% Set ticks in logarithmic scale but label them with original theta values
log_ticks = log10([min(distortion), max(distortion)]); % Logarithmic ticks
cb1.Ticks = log_ticks; % Use logarithmic spacing for ticks
cb1.TickLabels = arrayfun(@(x) sprintf('%.2f', 10^x), log_ticks, 'UniformOutput', false); % Convert back to original values

% Add labels and title
xlabel('k \cdot \theta', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Privacy Budget (\epsilon)', 'FontSize', 14, 'FontWeight', 'bold');
title('k \cdot \theta vs Privacy Budget (\epsilon) vs Distortion', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'XScale', 'log'); % Logarithmic x-axis
grid on;

% Customize colorbar label
cb1.Label.String = 'Distortion (Log Scale)';
cb1.Label.FontSize = 14;
cb1.Label.FontWeight = 'bold';

% gaussian_results = [];  % Store Gaussian results
% % sigma_log_values = logspace(log10(0.00001), log10(1.5), 20);
% % sigma_lin_values = linspace(0.1,150,10000); sigmad=[sigma_log_values,sigma_lin_values];
%  sigmad=linspace(0.1,10,200);
% % Step 2: Gaussian Noise Distortion
% lambda_max = 400;
% % gaussian_eps = zeros(20,1);
% % gaussian_dist=zeros(20,1);
% % gaussian_clip=zeros(20,1);
% % gaussian_q=zeros(20,1);
% % gaussian_sigma=zeros(20,1);
%  T_values=2000
%  delta_values=1/(2*67349);
% 
% q_values=512/67349;
% clip_values=[3];
% for idx_T=1:length(T_values)
% T=T_values(idx_T)
%     for idx_q = 1:length(q_values)
%      q=q_values(idx_q) 
%         for idx_delta=1:length(delta_values)
%     delta=delta_values(idx_delta)
% 
%     % 
%             for idx_clip = 1:length(clip_values)
%                  clip = clip_values(idx_clip);
%          for idx_sig=1:length(sigmad)
%             sigma=sigmad(idx_sig);
%              eps_check_gaussian =inf;
%             % Calculate sigma based on image formula
%                      % Loop over lambda up to lambda_max
%             for lambda = 1:lambda_max
%                 % Compute epsilon for Gaussian noise
% 
%                 % term1 = T * lambda / (2 * (sigma)^2);
%   lmbda1=lambda+1;              
%              % lmbda1=lambda      
% log_factorials = [0, cumsum(log(1:lmbda1))]; 
% 
% % Initialize the summation
% 
% 
%           % Initialize the summation
% S = 0;
% log_factorials = [0, cumsum(log(1:lmbda1))]; % Include log(0!) = 0 at the start
% 
% % Initialize the summation
% 
% 
% % Loop through all k values
% for kk = 0:lmbda1
%     % Compute binomial coefficient using logarithms
% 
%     % Compute the term
% 
%     % Compute binomial coefficient using logarithms
%     log_binom_coeff = log_factorials(lmbda1 + 1) - log_factorials(kk + 1) - log_factorials( lmbda1 - kk + 1);
% 
%     binom_coeff = exp(log_binom_coeff); % Convert back from log scale
% 
%     % Compute the term
%     exp_term = exp((kk^2 - kk) / (2 * sigma^2));
%     term = binom_coeff * (1 - q)^( lmbda1 - kk)* q^kk* exp_term;
% 
%     % Accumulate the result
%     S = S + term;
% end
% S=log(S)/(lambda);
% eps_check_gaussian1 = (T*S)+ log((lambda) / (lambda+1)) - ((log(delta) + log(lambda+1)) / (lambda));
% 
%                 % Calculate eps_check_gaussian and check if it meets eps_target
%                 % eps_check_gaussian = epsilon_gaussian(lambda) + (log(lambda / (1 + lambda)) - log(delta) + log(1 + lambda)) / lambda;
%                 % 
%               if eps_check_gaussian1>0
%                if eps_check_gaussian1 <= eps_check_gaussian 
%                    eps_check_gaussian=eps_check_gaussian1;
% 
%                     % Calculate distortion for Gaussian
%                   distortion_gaussian = sigma * sqrt(2 / pi)*clip;
%                     % if distortion_gaussian<distmin
%                         % distmin=distortion_gaussian;
%                     % Store the best Gaussian config for each eps_target
%                         gaussian_eps = eps_check_gaussian;
%                            gaussian_dist=distortion_gaussian;
%                           gaussian_clip=clip;
%                           gaussian_q=q;
%                           gaussian_sigma=sigma;
% 
%                 % end
%             end
%         end
%             end
%             gaussian_results = [gaussian_results; gaussian_eps, gaussian_q, gaussian_sigma, clip, gaussian_dist];
% 
%             end
%             end
% 
%      end
%     end
%  end
filtered_configs = gaussian_results (gaussian_results (:,1) < 10, :);

% Create scatter plot
gaussian_eps=Q(:,1);
gaussian_q=q;
gaussian_clip=clip;
gaussian_dist=Q(:,10)
gaussian_sigma=Q(:,10)./(clip*sqrt(2/pi));



% Create a figure
figure;
hold on;
grid on;

% Set up the figure
set(gca, 'FontSize', 12);
title('Comparison of PLRV and Gaussian Distortions', 'FontSize', 14);
xlabel('Distortion (d)', 'FontSize', 12);
ylabel('Privacy Budget (\epsilon)', 'FontSize', 12);

% Define colors and markers
colors = lines(2); % Two distinct colors for PLRV and Gaussian
markers = {'o', 'x'}; % Distinct markers for PLRV and Gaussian

% Process PLRV data
scatter(distortion, epsilon, 'o', 'DisplayName', 'PLRV');

hold on


% Process Gaussian data

    
    % Plot Gaussian data
    plot(gaussian_dist, gaussian_eps, 'LineWidth', 2, 'Color', colors(2, :),  'Marker', markers{2});

% Add a legend
legend('Location', 'best', 'FontSize', 10);

% Set x-axis to logarithmic scale
set(gca, 'XScale', 'log');

% Final touches
hold off;



% figure;
% % First subplot: Scatter plot with gaussian_eps as color
% subplot(1, 2, 1);
% plot(gaussian_sigma, gaussian_eps); % Marker size 40
% 
% xlabel('\sigma (gaussian\_sigma)');
% ylabel('Pivacy Budget (\epsilon)');
% 
% grid on;
% gaussian_results = sortrows(gaussian_results,1);
% % Filter for gaussian_eps < 10
% filter_idx = gaussian_eps < 10;
% 
% % Apply the filter
% filtered_sigma = gaussian_sigma(filter_idx);
% filtered_q = gaussian_q(filter_idx);
% filtered_dist = gaussian_dist(filter_idx);
% 
% % Second subplot: Scatter plot with gaussian_dist as color
% subplot(1, 2, 2);
% scatter(distortion, epsilon, 'o', 'DisplayName', 'PLRV'); % Scatter plot with log-scaled theta
% 
% % Add labels and title
% xlabel('Distortion (d)', 'FontSize', 14, 'FontWeight', 'bold');
% ylabel('Privacy Budget (\epsilon)', 'FontSize', 14, 'FontWeight', 'bold');
% title('k \cdot \theta vs Privacy Budget (\epsilon) vs Distortion', 'FontSize', 16, 'FontWeight', 'bold');
% set(gca, 'XScale', 'log'); % Logarithmic x-axis
% grid on;
% 
% xlabel('\sigma (gaussian\_sigma)');
% ylabel('Distortion (d)');
% 
% grid on;



% figure;
% 
% % First scatter plot
% 
% 
% 
% hold on;
% 
% % Second scatter plot
% scatter(filtered_dist, gaussian_eps, 'x', 'DisplayName', 'Gaussian'); % Marker size 40
% 
% % Add legend
% legend('Location', 'best', 'FontSize', 12, 'FontWeight', 'bold');
% 
% hold off;
