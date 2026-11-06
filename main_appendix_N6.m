function results = main_appendix_N6(mode)
%MAIN_APPENDIX_N6 Sensitivity/scalability experiments described in Appendix N.6.
% Put this file alongside the existing repository .m files, then run:
%   main_appendix_N6('full')        % all three sweeps and all sample sizes
% MATLAB R2022b or later is recommended.
%
% This driver reconstructs the published protocol with an explicit new seed;
% it does not contain the original random draws used for the manuscript tables.
% Estimation and metric definitions are delegated to the existing repository.

    if nargin < 1
        mode = 'full';
    end
    mode = validatestring(mode, {'smoke', 'population', 'full'});
    root_dir = fileparts(mfilename('fullpath'));
    addpath(root_dir);
    required = {'generate_multi_client_crl_data', ...
        'block_rq_decompose_all_clients', 'estimate_blocks_and_Qk', ...
        'estimate_global_ancestor_from_clients', 'evaluate_estimations', ...
        'aggregate_metric2', 'aggregate_metric3', 'tune_tau_repeated'};
    for i = 1:numel(required)
        assert(exist(required{i}, 'file') == 2, ...
            'Missing %s.m: put this driver in the repository root.', required{i});
    end

    cfg.mode = mode;
    cfg.noise_type = 'gaussian';  % change to 'laplace' for the other noise model
    cfg.range_G = [-1, 1];
    cfg.intervention_type = 'soft';
    % Existing generator semantics: REPLACE each nonzero incoming coefficient
    % and the target noise variance with independent Uniform[2,3] draws.
    cfg.range_A = [2, 3];
    cfg.range_Sigma = [2, 3];
    cfg.tau_list = 0.01:0.01:0.10;
    cfg.tuning_repetitions = 10;
    cfg.epsilon = 0.01;
    cfg.population_tau = 1e-4;
    cfg.population_epsilon = cfg.epsilon;
    cfg.sample_sizes = [1e4, 5e4, 1e5, 5e5, 1e6, Inf];
    cfg.evaluation_repetitions = 1; % increase for repeated independent trials

    sweeps = [repmat({'p'},4,1); repmat({'K'},4,1); repmat({'q'},4,1)];
    dimensions = [10;20;30;40;20;20;20;20;20;20;20;20];
    clients = [15;15;15;15;5;10;15;20;15;15;15;15];
    coverage = [.6;.6;.6;.6;.6;.6;.6;.6;.5;.6;.7;.8];
    settings = table(sweeps, dimensions, clients, coverage, ...
        'VariableNames', {'sweep', 'p', 'K', 'q'});
    if strcmp(mode, 'population')
        cfg.sample_sizes = Inf;
    elseif strcmp(mode, 'smoke')
        settings = table({'smoke'}, 5, 3, .7, ...
            'VariableNames', {'sweep', 'p', 'K', 'q'});
        cfg.sample_sizes = [200, Inf];
        cfg.tuning_repetitions = 1;
        cfg.tau_list = [0.01, 0.05];
    end

    output_dir = fullfile(root_dir, 'results_N6', mode);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    save(fullfile(output_dir, 'protocol.mat'), 'cfg', 'settings');
    rows = struct([]);
    weights = struct('f1', 1, 'metric2', 1, 'metric3', 1);

    for s = 1:height(settings)
        p = settings.p(s); K = settings.K(s); q = settings.q(s);
        for rep = 1:cfg.evaluation_repetitions
            % Use the same DAG at fixed p/rep. Client subsets are generated
            % without conditioning on global coverage or Assumption 3.
            graph_seed = cfg.seed + 1000 * rep + p;
            coverage_seed = cfg.seed + 100000 + 1000 * rep + p;
            model_seed = cfg.seed + 200000 + 1000 * rep + p;
            rng(graph_seed, 'twister');
            edges = triu(rand(p) < 3/p, 1);
            A = edges .* (2 * rand(p) - 1);
            rng(coverage_seed, 'twister');
            OM = double(rand(p, K)' < q);
            for k = 1:K
                % The existing generator requires a nonempty client subset.
                while ~any(OM(k,:))
                    OM(k,:) = double(rand(1,p) < q);
                end
            end
            nK = ceil(1.5 * sum(OM, 2))';

            for ni = 1:numel(cfg.sample_sizes)
                n = cfg.sample_sizes(ni);
                fprintf('\nN.6: sweep=%s, p=%d, K=%d, q=%.1f, n=%g, rep=%d\n', ...
                    settings.sweep{s}, p, K, q, n, rep);
                tuning = [];
                if isinf(n)
                    tau = cfg.population_tau;
                    epsilon = cfg.population_epsilon;
                    % The existing generator always draws samples. Use two
                    % dummy samples, then replace ALL estimated precisions.
                    generator_n = 2;
                else
                    rng(model_seed + 1000000 + ni, 'twister');
                    [tuning, tau] = tune_tau_repeated(cfg.tuning_repetitions, ...
                        A, OM, nK, cfg.range_G, cfg.noise_type, n, ...
                        cfg.intervention_type, cfg.range_Sigma, cfg.range_A, ...
                        0, cfg.tau_list, cfg.epsilon, weights);
                    epsilon = cfg.epsilon;
                    generator_n = n;
                end

                % The generator draws interventions and all mixing matrices
                % before samples: resetting this seed holds them fixed over n.
                rng(model_seed, 'twister');
                data = generate_multi_client_crl_data(A, OM, nK, ...
                    cfg.range_G, cfg.noise_type, generator_n, ...
                    cfg.intervention_type, cfg.range_Sigma, cfg.range_A, 0);
                data = block_rq_decompose_all_clients(data);
                if isinf(n)
                    for k = 1:K
                        for j = 1:numel(data.clients{k}.interventions)
                            data.clients{k}.interventions{j}.Hat_Theta_k = ...
                                data.clients{k}.interventions{j}.Theta_k;
                        end
                    end
                end

                % Time estimation/assembly only; exclude generation,
                % precision estimation, true RQ construction and evaluation.
                start_time = tic;
                data = estimate_blocks_and_Qk(data, tau);
                data = estimate_global_ancestor_from_clients(data);
                elapsed = toc(start_time);
                metrics = evaluate_estimations(data, epsilon);
                row = struct('sweep', settings.sweep{s}, 'p', p, 'K', K, ...
                    'q', q, 'replicate', rep, 'n', n, 'tau', tau, ...
                    'E1', metrics.global.f1, ...
                    'E2', aggregate_metric2(metrics), ...
                    'E3', aggregate_metric3(metrics), ...
                    'runtime_seconds', elapsed);
                if isempty(rows)
                    rows = row;
                else
                    rows(end+1) = row; %#ok<AGROW>
                end
                results = struct2table(rows);
                writetable(results, fullfile(output_dir, 'summary.csv'));
                filename = sprintf('%s_p%d_K%d_q%02d_rep%d_n%g.mat', ...
                    settings.sweep{s}, p, K, round(100*q), rep, n);
                save(fullfile(output_dir, filename), 'cfg', 'row', 'A', 'OM', ...
                    'nK', 'graph_seed', 'coverage_seed', 'model_seed', ...
                    'tuning', 'data', 'metrics', '-v7.3');
                disp(struct2table(row));
            end
        end
    end
    fprintf('\nResults saved to %s\n', output_dir);
end
