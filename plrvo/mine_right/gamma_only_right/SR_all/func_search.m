function [C] = func_search(C)

%% hyperparameters
% C = 0.1;
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

%% functions
M_u = @(k, theta, t) (1 - t .* theta).^(-k);
func_G = @(C, eta, k, theta) 0.5 * (M_u(k,theta,(eta-1)*C) + M_u(k,theta,-eta*C)) + (1/(2*(2*eta-1))) * (M_u(k,theta,(eta-1)*C) - M_u(k,theta,-eta*C));

%% output
% folderName = 'SearchResults';
% mkdir(folderName);
% warning('off', 'MATLAB:MKDIR:DirectoryExists');
folderName = sprintf('%s_%f', "SearchResults", C);
mkdir(folderName)
warning('off', 'MATLAB:MKDIR:DirectoryExists');

keys = dict.keys;
for idx = 1:numel(keys)
    dataset = keys{idx};
    params = dict(dataset);
    sample_size = params(1);
    batch_size = params(2);
    epoch = params(3);

    fprintf('Dataset: %s, Sample Size: %d, Batch Size: %d, Epoch: %d\n', dataset, sample_size, batch_size, epoch);

    % search for each task
    SSS = SSS_Finder(C, array_alpha, func_G);
    [S, param] = S_Finder(SSS, C, array_alpha, dataset, sample_size, batch_size, epoch, func_G);

    % save file
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filePath_S = sprintf('%s/%s_%d_%d_%d.%s.csv', folderName, dataset, sample_size, batch_size, epoch, timestamp);
    filePath_param = sprintf('%s/%s_%d_%d_%d.%s.json', folderName, dataset, sample_size, batch_size, epoch, timestamp);

    headers_S = {"G_k", "G_theta", "PLRV_distortion", "target_epsilon"};
    S = [headers_S; num2cell(S)];
    writecell(S, filePath_S);

    writestruct(param, filePath_param, 'FileType', 'json');

    fprintf('Results saved for Dataset: %s, Sample Size: %d, Batch Size: %d, Epoch: %d\n', dataset, sample_size, batch_size, epoch);

end
