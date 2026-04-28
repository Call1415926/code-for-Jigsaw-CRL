
clc; clear;

%% Parameter Setting

%%%setting 1
p = 5;
K = 5;
A = [0, 1, 0.7, 0, 0; 
     0, 0, 0, -1, 0;
     0, 0, 0, 1.3, -0.9;
     0, 0, 0, 0, 0.6;
     0, 0, 0, 0, 0];
OM = [1,0,1,1,1;
      1,1,0,1,1;
      1,1,1,1,0;
      1,1,1,0,0;
      0,1,1,0,1];
nK = [8, 6, 7, 6, 5]; %%%%n_k for each client k

% %%%setting 2
% p = 8;
% K = 4;
% A = [0,-0.8,0,1,0,0,0,0;
%      0,0,0.6,1.2,0,0,0,0;
%      0,0,0,0,0,0,0,1;
%      0,0,0,0,0,0,-0.9,0;
%      0,0,0,0,0,1.1,0,0;
%      0,0,0,0,0,0,0.7,0;
%      0,0,0,0,0,0,0,0;
%      0,0,0,0,0,0,0,0];
% OM = [0,1,1,1,0,0,1,1;
%       1,1,1,0,1,1,1,0;
%       1,1,0,1,1,1,0,1;
%       1,0,1,1,1,0,1,1];
% nK = [9, 10, 8,9];

%%%setting 3
% p = 10;
% K = 5;
% A = [0,0,-0.9,0,0,0,0,0,0,0;
%     0,0,1,0,0,0.7,0,0,0,0;
%     0,0,0,1.2,0,0,0,0,0,-0.8;
%     0,0,0,0,0,0,0,0,0,-1;
%     0,0,0,0,0,0,0,0,0,1.3;
%     0,0,0,0,0,0,0,1.1,0.9,0;
%     0,0,0,0,0,0,0,0,-0.5,0;
%     0,0,0,0,0,0,0,0,0,0;
%     0,0,0,0,0,0,0,0,0,0;
%     0,0,0,0,0,0,0,0,0,0];
% OM = [1,1,1,1,1,1,1,0,1,0;
%       0,0,1,0,1,1,1,1,0,1;
%       1,1,0,1,0,0,1,1,1,1;
%       1,0,1,1,1,1,1,0,0,1;
%       1,1,1,0,0,1,0,1,1,0];
% nK = [12, 10, 9, 11, 8];


is_n_infinity = 1;         %%% Indicates whether to use the infinite-sample setting (i.e., compute using the true theoretical precision matrix): 1 means yes, 0 means no.
range_G = [-1, 1];          %%%% range of element of Gk
noise_type = 'laplace';    %% type of noise: 'laplace' or 'gaussian'
intervention_type = 'soft'; %% type of intervention: 'soft' or 'hard'
range_Sigma = [2,3];        %%% range of variance of intervention
range_A = [2,3];            %%% rangee of inteervention value
record_data = 0;            %%%  ecord data

epsilon = 2e-2;             %%% When evaluating the metric, values below a certain threshold are treated as zero and considered outside the support set.

%%% choose hyperparameter tau
N = 10; 
tau_list = [0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.12,0.15,0.18,0.2,0.25];
weights = struct('f1', 1, 'metric2', 1, 'metric3', 1);

n_list = [1e3,5e3,1e4,3e4,5e4,8e4,1e5]%,3e5,4e5,5e5,7e5,8e5,9e5];   %%% 例如 1e4, 2e4, ..., 1e6
num_n = numel(n_list);

%%% Used to store the complete results for each n.
all_results = cell(num_n, 1);

%%% Used to store aggregated metrics in the CSV file.
summary_n = zeros(num_n, 1);
summary_best_tau = zeros(num_n, 1);
summary_f1 = zeros(num_n, 1);
summary_ancestor_error = zeros(num_n, 1);
summary_metric2 = zeros(num_n, 1);
summary_metric3 = zeros(num_n, 1);

%% Outer loop: run experiments for different values of n (number of samples).
for idx_n = 1:num_n
    n = n_list(idx_n);

    fprintf('\n====================================================\n');
    fprintf('Running experiment for n = %g  (%d / %d)\n', n, idx_n, num_n);
    fprintf('====================================================\n');

    %% 1) First tune the hyperparameter for the current n and select τ.
    if is_n_infinity == 0
        [summary_tau, best_tau] = tune_tau_repeated(N, A, OM, nK, range_G, noise_type, ...
            n, intervention_type, range_Sigma, range_A, record_data, tau_list, epsilon, weights);
        
        tau = best_tau;
    end

    %% 2) Generate the final data using the optimal τ and run the full pipeline.
    data = generate_multi_client_crl_data(A, OM, nK, range_G, noise_type, ...
        n, intervention_type, range_Sigma, range_A, record_data);

    %% Theoretical RQ decomposition.
    data = block_rq_decompose_all_clients(data);

    %% Set the hyperparameter τ for estimation.
    %%%% Replace the sample precision matrix with the theoretical precision matrix and compute the theoretical decomposition results.
    if is_n_infinity == 1
        for k = 1:K
            L = numel(data.clients{k}.interventions);
            for l = 1:L
                data.clients{k}.interventions{l}.Hat_Theta_k = data.clients{k}.interventions{l}.Theta_k;
            end
        end
        tau = 1e-4;
        epsilon = 1e-4;
    end 
    data = estimate_blocks_and_Qk(data, tau);

    %% 估计全局祖先关系
    data = estimate_global_ancestor_from_clients(data);

    %% Estimate the global ancestral relationships.
    metrics = evaluate_estimations(data, epsilon);

    f1_val = metrics.global.f1;
    ancestor_error_val = metrics.global.ancestor_error;
    metric2_val = aggregate_metric2(metrics);
    metric3_val = aggregate_metric3(metrics);

    fprintf('n = %g finished.\n', n);
    fprintf('  best tau         = %.6g\n', tau);
    fprintf('  global f1        = %.6f\n', f1_val);
    fprintf('  ancestor error   = %.6f\n', ancestor_error_val);
    fprintf('  mean metric2     = %.6f\n', metric2_val);
    fprintf('  mean metric3     = %.6f\n', metric3_val);

    %% 3) Store the complete results for the current n.
    result_struct = struct();
    result_struct.n = n;
    result_struct.best_tau = tau;
    if is_n_infinity == 0
        result_struct.summary_tau = summary_tau;
    end
    result_struct.data = data;
    result_struct.metrics = metrics;

    all_results{idx_n} = result_struct;

    %% 4) Store the summary metrics for the current n, which will later be written to the CSV file.
    summary_n(idx_n) = n;
    summary_best_tau(idx_n) = tau;
    summary_f1(idx_n) = f1_val;
    summary_ancestor_error(idx_n) = ancestor_error_val;
    summary_metric2(idx_n) = metric2_val;
    summary_metric3(idx_n) = metric3_val;

end

%% Final result table.
if is_n_infinity == 1
    summary_n = Inf * ones(num_n,1);
end

final_table = table(summary_n, summary_best_tau, summary_f1,  ...
                    summary_metric2, summary_metric3, ...
    'VariableNames', {'n', 'best_tau', 'f1', 'mean_metric2', 'mean_metric3'});

disp('==============================================');
disp('All experiments over n_list are finished.');
disp('==============================================');

final_table