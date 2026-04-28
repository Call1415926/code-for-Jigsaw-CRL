function [summary, best_tau] = tune_tau_repeated(N, A, OM, nK, range_G, noise_type, ...
    n, intervention_type, range_Sigma, range_A, record_data, tau_list, epsilon, weights)
% TUNE_TAU_REPEATED
% Perform repeated tuning of tau with multiple data generations.
% For each tau, it calculates the average performance over N repetitions.

% INPUT:
%   N                   : Number of repetitions (e.g., 20)
%   A, OM, nK, ...      : Parameters to generate the data
%   tau_list            : List of tau candidates
%   epsilon             : Small value for support extraction in metric 3
%   weights             : Weights for F1, Metric2, and Metric3

% OUTPUT:
%   summary             : A struct containing the summarized results
%   best_tau            : The best tau value

    % Initialize containers for storing results
    global_f1_all = zeros(N, numel(tau_list));
    mean_metric2_all = zeros(N, numel(tau_list));
    mean_metric3_all = zeros(N, numel(tau_list));

    all_metrics = cell(N, numel(tau_list));
    all_data = cell(N, numel(tau_list));

    % Repeat the process N times
    for n_iter = 1:N
        % Generate data for this iteration
        data_base = generate_multi_client_crl_data(A, OM, nK, range_G, noise_type, ...
            n, intervention_type, range_Sigma, range_A, record_data);

        % Theoretical RQ decomposition
        data_base = block_rq_decompose_all_clients(data_base);

        % Test each tau value in tau_list
        for it = 1:numel(tau_list)
            tau = tau_list(it);
            fprintf('Iteration %d / %d, tuning tau: %.6g (%d / %d)\n', n_iter, N, tau, it, numel(tau_list));

            % Perform estimation with the current tau
            data_tmp = data_base;
            data_tmp = clear_estimated_fields(data_tmp);

            data_tmp = estimate_blocks_and_Qk(data_tmp, tau);
            data_tmp = estimate_global_ancestor_from_clients(data_tmp);
            metrics_tmp = evaluate_estimations(data_tmp, epsilon);

            % Store the metrics
            global_f1_all(n_iter, it) = metrics_tmp.global.f1;
            mean_metric2_all(n_iter, it) = aggregate_metric2(metrics_tmp);
            mean_metric3_all(n_iter, it) = aggregate_metric3(metrics_tmp);

            all_metrics{n_iter, it} = metrics_tmp;
            all_data{n_iter, it} = data_tmp;
        end
    end

    % Average metrics over all repetitions
    global_f1_avg = mean(global_f1_all, 1);
    mean_metric2_avg = mean(mean_metric2_all, 1);
    mean_metric3_avg = mean(mean_metric3_all, 1);

    % Normalize metric2 and metric3 (min-max normalization)
    norm_metric2 = minmax_normalize(mean_metric2_avg);
    norm_metric3 = minmax_normalize(mean_metric3_avg);

    % Calculate the composite score for each tau
    score = weights.f1 * global_f1_avg ...
          - weights.metric2 * mean_metric2_avg ...
          - weights.metric3 * mean_metric3_avg;

    [~, best_index] = max(score);
    best_tau = tau_list(best_index);

    % Prepare summary table
    summary = struct();
    summary.tau_list = tau_list;
    summary.global_f1 = global_f1_avg;
    summary.mean_metric2 = mean_metric2_avg;
    summary.mean_metric3 = mean_metric3_avg;
    summary.norm_metric2 = norm_metric2;
    summary.norm_metric3 = norm_metric3;
    summary.score = score;
    summary.best_tau = best_tau;
    summary.best_index = best_index;

    % Create the result table for display
    result_table = table( ...
        tau_list(:), ...
        global_f1_avg(:), ...
        mean_metric2_avg(:), ...
        mean_metric3_avg(:), ...
        norm_metric2(:), ...
        norm_metric3(:), ...
        score(:), ...
        'VariableNames', { ...
            'tau', ...
            'global_f1', ...
            'mean_metric2', ...
            'mean_metric3', ...
            'norm_metric2', ...
            'norm_metric3', ...
            'score'});

    summary.table = result_table;

    fprintf('\nBest tau = %.6g\n', best_tau);
    fprintf('  global_f1   = %.6f\n', global_f1_avg(best_index));
    fprintf('  mean_metric2= %.6f\n', mean_metric2_avg(best_index));
    fprintf('  mean_metric3= %.6f\n', mean_metric3_avg(best_index));
    fprintf('  score       = %.6f\n', score(best_index));
end


%% ========================================================================
function data_out = clear_estimated_fields(data_in)
%CLEAR_ESTIMATED_FIELDS
% Removes estimated fields to reset the data for each iteration.

    data_out = data_in;

    if isfield(data_out, 'global_estimated')
        data_out = rmfield(data_out, 'global_estimated');
    end

    K = numel(data_out.clients);
    for k = 1:K
        if isfield(data_out.clients{k}, 'estimated_model')
            data_out.clients{k} = rmfield(data_out.clients{k}, 'estimated_model');
        end
    end
end

%% ========================================================================
function m2 = aggregate_metric2(metrics)
% Aggregates metric 2 (block subspace error) across clients.

    K = numel(metrics.clients);
    vals = nan(K,1);

    for k = 1:K
        vals(k) = metrics.clients{k}.metric2_mean_block_subspace_error;
    end

    m2 = mean(vals, 'omitnan');
end

%% ========================================================================
function m3 = aggregate_metric3(metrics)
% Aggregates metric 3 (leakage error) across clients.

    K = numel(metrics.clients);
    vals = nan(K,1);

    for k = 1:K
        vals(k) = metrics.clients{k}.metric3_leakage_error;
    end

    m3 = mean(vals, 'omitnan');
end

%% ========================================================================
function x_norm = minmax_normalize(x)
% Min-max normalization to [0, 1].

    xmin = min(x);
    xmax = max(x);

    if abs(xmax - xmin) <= eps
        x_norm = zeros(size(x));
    else
        x_norm = (x - xmin) / (xmax - xmin);
    end
end