T_values = [linspace(1, 10^2, 4) linspace(1000, 10^4, 6)];
q_values = logspace(-6, -2, 5);
delta_values = logspace(-8, -4, 5);

clip_values = [0.1 0.5 1 2 3];
% Q=Q(Q(:,8)<2000,:);

% Q=Q(Q(:,1)<5,:);
Q=G;

T_sample = [T_values(1), T_values(3), T_values(5)];

for clip_idx = 2:4
    clip_filtered = Q(Q(:, 4) == clip_values(clip_idx), :);

    % Accounting Plots (1-a)
    figure;
    sgtitle(sprintf('Accounting Plots: Clip = %.1f', clip_values(clip_idx)));

    for delta_idx = 1:length(delta_values)
        subplot(2, 3, delta_idx); % Reserve subplot space

        % Filter data for the current delta
        delta_filtered = clip_filtered(clip_filtered(:, 3) == delta_values(delta_idx), :);

        % Handle empty or invalid data
        if isempty(delta_filtered)
            text(0.5, 0.5, sprintf('No data for delta = %.1e', delta_values(delta_idx)), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'red');
            axis off; % Hide axes for placeholder
            continue;
        end

        if any(delta_filtered(:, 1) <= 0)
            text(0.5, 0.5, 'Non-positive values detected', ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'red');
            axis off; % Hide axes for placeholder
            continue;
        end

        % Apply logarithmic transformation
        log_colors = log10(delta_filtered(:, 1));

        % Create scatter plot
        scatter(delta_filtered(:, 8), log10(delta_filtered(:, 5)), 50, log_colors, 'filled');

        % Customize colorbar
        cb = colorbar;
        caxis([min(log_colors), max(log_colors)]);
        valid_values = delta_filtered(:, 1);
        valid_values = valid_values(valid_values > 0 & isfinite(valid_values)); % Retain finite positive values

        if isempty(valid_values)
            warning('No valid values for colorbar ticks.');
            continue;
        end

        linear_ticks = logspace(log10(min(valid_values)), log10(max(valid_values)), 7);
        log_ticks = log10(linear_ticks);
        formatted_ticks = arrayfun(@(x) sprintf('%.2f', x), linear_ticks, 'UniformOutput', false);
        set(cb, 'Ticks', log_ticks, 'TickLabels', formatted_ticks);

        % Add labels and title
        title(sprintf('Delta = 10^{-%d}', -log10(delta_values(delta_idx))));
        xlabel('Rounds (T)');
        ylabel('log(q)');
        ylabel(cb, '\epsilon');
        grid on;
    end
end

% 
%     % Accounting Plots (1-b)
% 
% %    % Accounting Plots (1-b)
for T_idx = 1:3
    T_filtered = clip_filtered(clip_filtered(:, 8) == T_sample(T_idx), :);

    % Apply logarithmic transformation to T_filtered(:, 1)
    log_colors = log10(T_filtered(:, 1)); % Logarithmic spacing for color

    subplot(2, 3, T_idx + 3);
    scatter(log10(T_filtered(:, 3)), log10(T_filtered(:, 5)), 50, log_colors, 'filled');

    % Customize colorbar for logarithmic spacing but with linear ticks
    cb1 = colorbar;
    caxis([min(log_colors), max(log_colors)]); % Set the color axis range
valid_values = T_filtered(:, 1);
valid_values = valid_values(isfinite(valid_values)); % Retain only finite values

% Check if valid_values is non-empty
if isempty(valid_values)
    warning('All values in delta_filtered(:, 1) are Inf or invalid.');
    continue; % Skip this iteration
end
    % Define meaningful tick points
  linear_ticks=logspace(log10(0.01), log10(max(valid_values)),6);

    log_ticks = log10(linear_ticks); % Convert to log scale

         % Truncate to 2 decimal places
formatted_ticks = arrayfun(@(x) sprintf('%.2f', x), linear_ticks, 'UniformOutput', false);
set(cb1, 'Ticks', log_ticks, 'TickLabels', formatted_ticks); % Apply truncated tick labels
% Set x-axis ticks
XX=(-8:-2);
xticks(XX); % Use -8, -7, -6 as ticks
xticklabels(arrayfun(@(x) sprintf('10^{%d}', x), XX, 'UniformOutput', false)); % Format ticks as 10^(-8), etc.
    % Add title and labels
    title(sprintf('T = %.1f', T_sample(T_idx)));
    xlabel('log(\delta)');
    ylabel('log(q)');
    ylabel(cb1, '\epsilon');


end


% Group 2: Optimized Search Space
figure;
clip_filtered = Q(Q(:, 4) == clip_values(1), :);

% Filter for specific q value
q_filtered = clip_filtered(abs(log10(clip_filtered(:, 5)) - log10(0.001)) < 1e-6, :);

% Iterate over delta values
for delta_idx = 1:length(delta_values)
    % Filter data for the current delta value
    delta_filtered = q_filtered(abs(log10(q_filtered(:, 3)) - log10(delta_values(delta_idx))) < 1e-6, :);

    if isempty(delta_filtered)
        % Skip if no data is available for the current delta
        warning('No data for delta = %e', delta_values(delta_idx));
        continue;
    end

    % Optimized Scale (k)
    subplot(2, 3, delta_idx); % Adjust subplot index
    scatter(delta_filtered(:, 8), delta_filtered(:, 1), 50, delta_filtered(:, 6), 'filled'); % Scale (k) as color
    title(sprintf('\\delta = 10^{-%d}', -log10(delta_values(delta_idx))));
    xlabel('Rounds (T)');
    ylabel('\epsilon');
    cb = colorbar;
    if ~isempty(delta_filtered(:, 6))
        caxis([min(delta_filtered(:, 6)), max(delta_filtered(:, 6))]); % Adjust color scale
    else
        warning('No valid values for k. Skipping caxis adjustment.');
    end
    ylabel(cb, 'Optimal Scale (k)');

    % Optimized Shape (\theta)
    subplot(2, 3, delta_idx + 3); % Adjust subplot index for second row
    scatter(delta_filtered(:, 8), delta_filtered(:, 1), 50, delta_filtered(:, 7), 'filled'); % Shape (\theta) as color
    title(sprintf('\\delta = 10^{-%d}', -log10(delta_values(delta_idx))));
    xlabel('Rounds (T)');
    ylabel('\epsilon');
    cb = colorbar;
    if ~isempty(delta_filtered(:, 7))
        caxis([min(delta_filtered(:, 7)), max(delta_filtered(:, 7))]); % Adjust color scale
    else
        warning('No valid values for \theta. Skipping caxis adjustment.');
    end
    ylabel(cb, 'Optimal Shape (\theta)');
end
T_sample = [T_values(2), T_values(4), T_values(8)];
   figure;
Group 3: Privacy-Utility Trade-Off
for clip_idx = 1:3
    figure;
    clip_filtered = Q(Q(:, 4) == clip_values(clip_idx) & Q(:, 3) == 1e-6, :);

    Best Distortion for q values

    sgtitle(sprintf('Privacy-Utility Trade-Off: Clip = %.1f', clip_values(clip_idx)));
    for q_idx = 1:3
        q_filtered = clip_filtered(clip_filtered(:, 5)==q_values(q_idx), :);
        subplot(2, 3, q_idx);
        scatter(q_filtered(:, 1), q_filtered(:, 2), 50, q_filtered(:, 8), 'filled');
        title(sprintf('q = %.1e', q_values(q_idx)));
        xlabel('\epsilon');
        ylabel('Distortion (d)');
        colorbar;
        ylabel(colorbar, 'Rounds (T)');
    end

    Best Distortion for T values
    for T_idx = 1:length(T_sample)
        T_filtered = clip_filtered(clip_filtered(:, 8)==T_sample(T_idx), :);
        subplot(2, 3, T_idx + 3);
        scatter(T_filtered(:, 1), T_filtered(:, 2), 50, T_filtered(:, 5), 'filled');
        title(sprintf('T = %.1f', T_values(T_idx)));
        xlabel('\epsilon');
        ylabel('Distortion (d)');
        colorbar;
        ylabel(colorbar, 'Clip (C)');
    end
end
