function data = generate_multi_client_crl_data(A, OM, nK, range_G, noise_type, n, intervention_type, range_Sigma, range_A,record_data)
%GENERATE_MULTI_CLIENT_CRL_DATA
% Generate synthetic data for the multi-client CRL model described by user.
%
% INPUTS
%   A                 : p x p strict upper triangular global DAG matrix
%   OM                : K x p binary matrix; row k indicates O_k
%   nK                : 1 x K vector; nK(k) = dimension of X_k
%   range_G           : [a, b], entry-wise range of G_k
%   noise_type        : 'gaussian' or 'laplace'
%   n                 : number of samples per client per intervention
%   intervention_type : 'soft' or 'hard'
%   range_Sigma       : [c, d], range of intervened global Sigma(j,j)
%   range_A           : [e, f], range of modified nonzero entries in row j under soft intervention
%   record_data       : 0-1 variable, 1 means record X samples and Z samples 
%
% OUTPUT
%   data : struct containing all generated objects
%
% MATLAB version target: R2022b

    %% =========================
    % 0. Basic checks
    % ==========================
    validate_inputs(A, OM, nK, range_G, noise_type, n, intervention_type, range_Sigma, range_A);

    p = size(A, 1);
    K = size(OM, 1);

    %% =========================
    % 1. Global observational and intervention models
    % ==========================
    Sigma0 = eye(p);

    % Global intervention indexing:
    % j = 0 means observational
    % j = 1,...,p means intervention on global node j
    %
    % We store these in cell arrays of length p+1, where:
    %   idx = j+1
    A_global = cell(p+1, 1);
    Sigma_global = cell(p+1, 1);
    target_node = zeros(p+1, 1);   % target_node(1)=0, target_node(j+1)=j

    A_global{1} = A;
    Sigma_global{1} = Sigma0;
    target_node(1) = 0;

    for j = 1:p
        [Aj, Sigmaj] = build_global_intervention(A, Sigma0, j, intervention_type, range_Sigma, range_A);
        A_global{j+1} = Aj;
        Sigma_global{j+1} = Sigmaj;
        target_node(j+1) = j;
    end

    %% =========================
    % 2. Generate client-specific G_k, H_k
    % ==========================
    clients = cell(K, 1);

    for k = 1:K
        Ok = find(OM(k, :) == 1);
        Mk = find(OM(k, :) == 0);
        pk = numel(Ok);
        nk = nK(k);

        if nk < pk
            error('Client %d has nk=%d < |Ok|=%d. Cannot generate full-column-rank Gk.', k, nk, pk);
        end

        Gk = sample_full_column_rank_matrix(nk, pk, range_G);
        Hk = (Gk' * Gk) \ Gk';

        client = struct();
        client.k = k;
        client.Ok = Ok(:)';
        client.Mk = Mk(:)';
        client.pk = pk;
        client.nk = nk;
        client.G = Gk;
        client.H = Hk;

        % valid interventions for client k:
        % j = 0 and j in Ok
        valid_j = [0, Ok(:)'];
        client.valid_j = valid_j;

        % store observational block structure later
        client.block_info = [];

        % intervention records
        client.interventions = cell(numel(valid_j), 1);

        clients{k} = client;
    end

    %% =========================
    % 3. Per-client, per-valid-intervention computations and sampling
    % ==========================
    for k = 1:K
        client = clients{k};
        Ok = client.Ok;
        Mk = client.Mk;
        Gk = client.G;
        Hk = client.H;
        valid_j = client.valid_j;

        for t = 1:numel(valid_j)
            j = valid_j(t);   % global intervention index

            Aj_global = A_global{j+1};
            Sigmaj_global = Sigma_global{j+1};

            % Marginalized / induced client-level model
            [Akj, Sigmakj] = compute_client_model(Aj_global, Sigmaj_global, Ok, Mk);

            % Theoretical Theta_k^(j)
            Ik = eye(numel(Ok));
            Theta_kj = Hk' * (Ik - Akj)' * (Sigmakj \ (Ik - Akj)) * Hk;

            % Theoretical covariance of Z_Ok^(j)
            CovZ_kj = ((Ik - Akj) \ Sigmakj) / (Ik - Akj)';

            % Generate Z samples: pk x n
            Z_samples = generate_zero_mean_samples(CovZ_kj, n, noise_type);

            % Generate X samples: nk x n
            X_samples = Gk * Z_samples;

            % Empirical covariance of X samples (centered)
            Xc = X_samples - mean(X_samples, 2);
            if n == 1
                warning('n=1 for client %d intervention j=%d. Sample covariance is undefined; returning zeros.', k, j);
                Sx = zeros(size(X_samples, 1));
            else
                Sx = (Xc * Xc.') / (n - 1);
            end

            % Empirical pseudo-precision
            Hat_Theta = pinv(Sx);

            rec = struct();
            rec.j = j;
            rec.target_node = j; % j=0 means observational, otherwise target node = j
            rec.A_global = Aj_global;
            rec.Sigma_global = Sigmaj_global;
            rec.A_k = Akj;
            rec.Sigma_k = Sigmakj;
            rec.Theta_k = Theta_kj;
            rec.CovZ_k = CovZ_kj;
            if record_data == 1
                rec.Z_samples = Z_samples;
                rec.X_samples = X_samples;
            end 
            rec.sample_cov_X = Sx;
            rec.Hat_Theta_k = Hat_Theta;

            client.interventions{t} = rec;

            % Build block structure using observational model only (j=0)
            if j == 0
                block_info = compute_block_structure(Akj, Sigmakj, Ok);
                client.block_info = block_info;
            end
        end

        clients{k} = client;
    end

    %% =========================
    % 4. Final output
    % ==========================
    data = struct();
    data.description = 'Synthetic data for multi-client CRL model';
    data.input = struct( ...
        'A', A, ...
        'OM', OM, ...
        'nK', nK, ...
        'range_G', range_G, ...
        'noise_type', noise_type, ...
        'n', n, ...
        'intervention_type', intervention_type, ...
        'range_Sigma', range_Sigma, ...
        'range_A', range_A);

    data.global = struct();
    data.global.p = p;
    data.global.K = K;
    data.global.A_observational = A;
    data.global.Sigma_observational = Sigma0;
    data.global.A_interventions = A_global;        % cell, j=0..p stored at j+1
    data.global.Sigma_interventions = Sigma_global;
    data.global.target_node = target_node;         % target_node(j+1)=j
    data.global.intervention_index_meaning = ...
        'Cell index idx corresponds to global intervention j = idx-1; j=0 means observational';

    data.clients = clients;

end

%% ========================================================================
%  Helper functions
% ========================================================================

function validate_inputs(A, OM, nK, range_G, noise_type, n, intervention_type, range_Sigma, range_A)

    if ~ismatrix(A) || size(A,1) ~= size(A,2)
        error('A must be a square matrix.');
    end

    p = size(A,1);

    if ~isequal(size(OM,2), p)
        error('OM must be K x p where p = size(A,1).');
    end

    if ~isnumeric(OM) || any(~ismember(OM(:), [0,1]))
        error('OM must be a binary 0-1 matrix.');
    end

    if any(sum(OM, 2) == 0)
        error('Each client must observe at least one latent variable; every row of OM must contain at least one 1.');
    end

    K = size(OM,1);

    if ~isvector(nK) || numel(nK) ~= K
        error('nK must be a 1 x K or K x 1 vector with K = size(OM,1).');
    end
    nK = nK(:)';

    if any(nK <= 0) || any(abs(nK - round(nK)) > 0)
        error('All entries in nK must be positive integers.');
    end

    if ~isnumeric(range_G) || numel(range_G) ~= 2 || range_G(1) > range_G(2)
        error('range_G must be [a,b] with a <= b.');
    end

    if ~ischar(noise_type) && ~isstring(noise_type)
        error('noise_type must be a string.');
    end
    noise_type = lower(string(noise_type));
    if ~ismember(noise_type, ["gaussian", "laplace"])
        error('noise_type must be ''gaussian'' or ''laplace''.');
    end

    if ~isscalar(n) || n <= 0 || abs(n - round(n)) > 0
        error('n must be a positive integer.');
    end

    if ~ischar(intervention_type) && ~isstring(intervention_type)
        error('intervention_type must be a string.');
    end
    intervention_type = lower(string(intervention_type));
    if ~ismember(intervention_type, ["soft", "hard"])
        error('intervention_type must be ''soft'' or ''hard''.');
    end

    if ~isnumeric(range_Sigma) || numel(range_Sigma) ~= 2 || range_Sigma(1) > range_Sigma(2)
        error('range_Sigma must be [c,d] with c <= d.');
    end
    if range_Sigma(1) <= 0
        error('range_Sigma must contain strictly positive values because Sigma diagonal entries must be positive.');
    end

    if ~isnumeric(range_A) || numel(range_A) ~= 2 || range_A(1) > range_A(2)
        error('range_A must be [e,f] with e <= f.');
    end

    % strict upper triangular check
    if any(abs(diag(A)) > 0)
        error('A must have zero diagonal.');
    end
    if any(any(tril(A,0) ~= 0))
        error('A must be strictly upper triangular.');
    end

    % full-column-rank feasibility for each client
    for k = 1:K
        pk = sum(OM(k,:));
        if nK(k) < pk
            error('Client %d has nk=%d < |Ok|=%d, so full-column-rank Gk is impossible.', k, nK(k), pk);
        end
    end

end

function [Aj, Sigmaj] = build_global_intervention(A, Sigma0, j, intervention_type, range_Sigma, range_A)
% Build global intervened model for intervention on node j.

    p = size(A,1);
    Aj = A;
    Sigmaj = Sigma0;

    % Change Sigma(j,j)
    Sigmaj(j,j) = unifrnd_scalar(range_Sigma(1), range_Sigma(2));

    intervention_type = lower(string(intervention_type));

    switch intervention_type
        case "soft"
            nz_idx = find(A(j,:) ~= 0);  % only modify original nonzero entries
            if ~isempty(nz_idx)
                Aj(j, nz_idx) = range_A(1) + (range_A(2) - range_A(1)) * rand(1, numel(nz_idx));
            end

        case "hard"
            nz_idx = find(A(j,:) ~= 0);  % only original nonzero positions are changed, all set to zero
            if ~isempty(nz_idx)
                Aj(j, nz_idx) = 0;
            end

        otherwise
            error('Unknown intervention_type.');
    end

    % Safety check: Aj should remain strictly upper triangular
    if any(abs(diag(Aj)) > 0) || any(any(tril(Aj,0) ~= 0))
        error('Intervened Aj is not strictly upper triangular. Check implementation.');
    end

    % Since A is strictly upper triangular, (I-A) is always invertible.
    % No extra stability check is needed here.
    %#ok<NASGU>
end

function [Ak, Sigmak] = compute_client_model(Aglobal, Sigmaglobal, Ok, Mk)
% Compute client-level induced model after marginalizing over Mk.

    pk = numel(Ok);

    if isempty(Mk)
        Ak = Aglobal(Ok, Ok);
        Sigmak = Sigmaglobal(Ok, Ok);
        return;
    end

    Aoo = Aglobal(Ok, Ok);
    Aom = Aglobal(Ok, Mk);
    Amo = Aglobal(Mk, Ok);
    Amm = Aglobal(Mk, Mk);

    Soo = Sigmaglobal(Ok, Ok);
    Smm = Sigmaglobal(Mk, Mk);

    Im = eye(numel(Mk));
    T = (Im - Amm) \ eye(numel(Mk));   % = inv(I - Amm)

    Ak = Aoo + Aom * T * Amo;
    Sigmak = Soo + Aom * T * Smm * T' * Aom';

    % small symmetrization for numerical stability
    Sigmak = (Sigmak + Sigmak') / 2;

    % optional PSD safeguard
    if pk > 0
        min_eig = min(eig(Sigmak));
        if min_eig <= 0
            % numerical repair if needed
            Sigmak = Sigmak + (abs(min_eig) + 1e-10) * eye(pk);
        end
    end

end

function G = sample_full_column_rank_matrix(nk, pk, range_G)
% Sample nk x pk matrix with entries in [a,b] until full column rank.

    a = range_G(1);
    b = range_G(2);

    max_try = 10000;
    for it = 1:max_try
        G = a + (b - a) * rand(nk, pk);
        if rank(G) == pk
            return;
        end
    end
    error('Failed to generate a full-column-rank G matrix after %d attempts.', max_try);

end

function X = generate_zero_mean_samples(CovMat, n, noise_type)
% Generate d x n samples with mean 0 and target covariance CovMat.
% For Gaussian: exact covariance model by linear transform of standard normal.
% For Laplace: independent variance-1 Laplace entries, then linear transform.

    d = size(CovMat, 1);

    % Symmetrize and compute square root robustly
    CovMat = (CovMat + CovMat') / 2;
    [V, D] = eig(CovMat);
    lam = diag(D);
    lam(lam < 0) = 0;
    SqrtCov = V * diag(sqrt(lam)) * V';

    noise_type = lower(string(noise_type));

    switch noise_type
        case "gaussian"
            U = randn(d, n);   % Cov(U)=I

        case "laplace"
            U = sample_standard_laplace(d, n);  % each entry variance 1, independent

        otherwise
            error('Unknown noise_type.');
    end

    X = SqrtCov * U;

    % optional recenter to very small numerical mean
    X = X - mean(X, 2);

end

function U = sample_standard_laplace(d, n)
% Generate d x n independent Laplace(0,b) entries with variance 1.
% For Laplace(0,b), variance = 2 b^2, so choose b = 1/sqrt(2).

    b = 1 / sqrt(2);
    R = rand(d, n) - 0.5;

    U = zeros(d, n);

    idx1 = (R < 0);
    idx2 = ~idx1;

    % inverse CDF of Laplace
    U(idx1) = b * log(1 + 2 * R(idx1));
    U(idx2) = -b * log(1 - 2 * R(idx2));

end

function block_info = compute_block_structure(Ak, Sigmak, Ok_global)
% Compute client block structure and block-level ancestor / parent structure
% using the observational model (typically j=0).
%
% Conventions:
%   Ak(u,v) ~= 0 means v -> u.
%   So v is parent/ancestor of u.
%
% Reachability for block definition:
%   You said a path may use ancestor relation or bidirected edges.
%   We implement this as a directed mixed graph with:
%     - descendant -> ancestor edges (i -> a if a is ancestor of i)
%     - bidirected edges represented by two directed edges
%   Then blocks are SCCs of this directed graph.

    pk = size(Ak,1);

    % 1) Ancestor relation from Ak
    parent_adj = double(Ak ~= 0);   % parent_adj(u,v)=1 means v -> u in SEM
    anc = compute_ancestor_matrix_from_parent_adj(parent_adj);
    % anc(u,v)=1 means v is ancestor of u

    % 2) Bidirected relation from Sigmak off-diagonal nonzeros
    bidir = double(abs(Sigmak) > 1e-12);
    bidir(1:pk+1:end) = 0;  % zero diagonal

    % 3) Directed graph for SCC block definition
    % descendant -> ancestor edges:
    % if v ancestor of u, then edge u -> v
    reach_adj = double(anc > 0)';  % reach_adj(i,j)=1 if i -> j by ancestor-step
    % add bidirected edges both directions
    reach_adj = double((reach_adj + bidir) > 0);
    reach_adj(1:pk+1:end) = 0;

    G = digraph(reach_adj);
    comp = conncomp(G, 'Type', 'strong');  % SCC labels, 1 x pk
    num_blocks = max(comp);

    blocks_local = cell(num_blocks, 1);
    blocks_global = cell(num_blocks, 1);
    for b = 1:num_blocks
        idx = find(comp == b);
        blocks_local{b} = idx(:)';
        blocks_global{b} = Ok_global(idx);
    end

    % 4) Block-level ancestor matrix
    % block b1 is ancestor of block b2 iff
    % exists u in b1, v in b2 such that u ancestor of v
    % Since anc(v,u)=1 means u ancestor of v.
    block_ancestor = zeros(num_blocks);

    for b1 = 1:num_blocks
        U = blocks_local{b1};
        for b2 = 1:num_blocks
            if b1 == b2
                continue;
            end
            V = blocks_local{b2};
            flag = false;
            for uu = U
                for vv = V
                    if anc(vv, uu) > 0
                        flag = true;
                        break;
                    end
                end
                if flag
                    break;
                end
            end
            block_ancestor(b1, b2) = flag;
        end
    end

    % 5) Block-level parent matrix (cover relation / Hasse relation)
    block_parent = zeros(num_blocks);
    for b1 = 1:num_blocks
        for b2 = 1:num_blocks
            if b1 == b2 || block_ancestor(b1, b2) == 0
                continue;
            end

            has_intermediate = false;
            for bm = 1:num_blocks
                if bm == b1 || bm == b2
                    continue;
                end
                % bm is descendant of b1 and ancestor of b2
                if block_ancestor(b1, bm) == 1 && block_ancestor(bm, b2) == 1
                    has_intermediate = true;
                    break;
                end
            end

            if ~has_intermediate
                block_parent(b1, b2) = 1;
            end
        end
    end

    block_info = struct();
    block_info.num_blocks = num_blocks;
    block_info.blocks_local = blocks_local;
    block_info.blocks_global = blocks_global;
    block_info.node_to_block = comp;
    block_info.parent_adj_local = parent_adj;
    block_info.ancestor_matrix_local = anc;
    block_info.bidirected_matrix_local = bidir;
    block_info.reachability_graph_for_blocks = reach_adj;
    block_info.block_ancestor_matrix = block_ancestor;
    block_info.block_parent_matrix = block_parent;

end

function anc = compute_ancestor_matrix_from_parent_adj(parent_adj)
% parent_adj(u,v)=1 means v -> u (v is parent of u)
% Return anc(u,v)=1 if v is ancestor of u (including possibly via multi-step),
% excluding self-ancestor on diagonal.

    pk = size(parent_adj, 1);

    % Build standard directed adjacency parent -> child
    % if v -> u, then edge v -> u
    A_dir = parent_adj';  % A_dir(v,u)=1

    anc = zeros(pk);

    for s = 1:pk
        visited = false(1, pk);
        queue = s;
        visited(s) = true;

        while ~isempty(queue)
            cur = queue(1);
            queue(1) = [];

            children = find(A_dir(cur, :) ~= 0);
            for ch = children
                if ~visited(ch)
                    visited(ch) = true;
                    queue(end+1) = ch; %#ok<AGROW>
                end
            end
        end

        % visited nodes are descendants of s, excluding itself if wanted
        visited(s) = false;
        desc = find(visited);
        % s is ancestor of each node in desc, i.e., anc(desc, s)=1
        anc(desc, s) = 1;
    end

end

function x = unifrnd_scalar(a, b)
    x = a + (b - a) * rand();
end