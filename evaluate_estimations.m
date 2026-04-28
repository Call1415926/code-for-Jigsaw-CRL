function metrics = evaluate_estimations(data, epsilon)
%EVALUATE_ESTIMATIONS
% Evaluate estimated results using three metrics:
%
% Metric 1 (global):
%   Error between estimated global ancestor matrix and true global ancestor matrix.
%
% Metric 2 (per client, per estimated block):
%   Subspace error between estimated block subspace and the true Q_k block subspace.
%
% Metric 3 (per client):
%   Structural leakage error of aligned \hat Q_k * G_k relative to the TRUE
%   block-level ancestor structure from generated data.
%
% IMPORTANT CONVENTION:
%   All TRUE quantities are indexed by the natural Z numbering.
%   All ESTIMATED quantities are indexed by intervention labels.
%   Therefore, BEFORE evaluation, estimated quantities are aligned to true Z numbering.
%
% INPUT
%   data    : struct from the full pipeline
%   epsilon : threshold for support extraction in Metric 3
%
% OUTPUT
%   metrics : struct with fields
%       .global
%       .clients{k}
%
% Required fields expected in data:
%   data.input.A
%   data.global.target_node               (or identity mapping is assumed)
%   data.global_estimated.global_ancestor_matrix
%   data.clients{k}.Ok
%   data.clients{k}.G
%   data.clients{k}.block_info
%   data.clients{k}.block_rq.Qk
%   data.clients{k}.estimated_model.blocks
%   data.clients{k}.estimated_model.Qhat

    if nargin < 2
        epsilon = 1e-8;
    end

    validate_evaluation_inputs(data);

    p = size(data.input.A, 1);
    K = numel(data.clients);

    % ------------------------------------------------------------
    % 0. Global label -> true Z-node mapping
    % ------------------------------------------------------------
    label_to_true = get_label_to_true_mapping(data, p);

    % ------------------------------------------------------------
    % 1. Metric 1: global ancestor structure error
    % ------------------------------------------------------------
    A_true = data.input.A;
    Anc_true = true_ancestor_matrix_from_A(A_true);  % convention: (child, ancestor)=1

    Anc_est_label = data.global_estimated.global_ancestor_matrix;
    Anc_est_true = align_square_matrix_label_to_true(Anc_est_label, label_to_true, p);

    global_err = offdiag_hamming_error(Anc_true, Anc_est_true);
    [prec, rec, f1] = binary_prf1_offdiag(Anc_true, Anc_est_true);

    metrics = struct();
    metrics.global = struct();
    metrics.global.true_ancestor_matrix = Anc_true;
    metrics.global.estimated_ancestor_matrix_aligned = Anc_est_true;
    metrics.global.ancestor_error = global_err;
    metrics.global.precision = prec;
    metrics.global.recall = rec;
    metrics.global.f1 = f1;
    metrics.global.convention = '(i,j)=1 means j is an ancestor of i';

    % ------------------------------------------------------------
    % 2. Metric 2 and 3: per client
    % ------------------------------------------------------------
    metrics.clients = cell(K, 1);

    for k = 1:K
        client = data.clients{k};

        Ok = client.Ok(:)';                 % true node labels in local order
        pk = numel(Ok);
        Gk = client.G;                      % nk x pk
        Qk_true = client.block_rq.Qk;       % pk x nk, TRUE block-RQ Qk
        est = client.estimated_model;       % estimated blocks, Qhat

        % --- align estimated Qhat rows to TRUE local node order of Ok
        [Qhat_aligned_local, row_block_id_local, est_blocks_true_nodes] = ...
            align_estimated_Qhat_to_true_local(est, Ok, label_to_true);

        % ---------------- Metric 2: block subspace error ----------------
        block_subspace_errors = nan(numel(est.blocks), 1);
        block_details = cell(numel(est.blocks), 1);

        for b = 1:numel(est.blocks)
            est_block_labels = est.blocks{b};
            est_block_true_nodes = map_labels_to_true_nodes(est_block_labels, label_to_true);

            % convert true node labels -> local row indices in this client
            local_idx = nodes_to_local_indices(est_block_true_nodes, Ok);

            Qhat_block = est.Qhat(block_rows_from_blocks(est.blocks, b), :);
            Qtrue_block = Qk_true(local_idx, :);

            block_subspace_errors(b) = subspace_projection_error(Qhat_block, Qtrue_block);

            block_details{b} = struct( ...
                'estimated_block_labels', est_block_labels, ...
                'estimated_block_true_nodes', est_block_true_nodes, ...
                'local_row_indices_in_client', local_idx, ...
                'subspace_error', block_subspace_errors(b));
        end

        % ---------------- Metric 3: structural leakage ----------------
        Mhat = Qhat_aligned_local * Gk;     % pk x pk

        support_est = abs(Mhat) > epsilon;
        support_true_allowed = true_support_mask_from_block_structure(client.block_info);

        leakage = structural_leakage_rate(support_est, support_true_allowed);
        mismatch = support_mismatch_rate(support_est, support_true_allowed);

        client_metrics = struct();
        client_metrics.Ok_true_nodes = Ok;
        client_metrics.estimated_blocks_true_nodes = est_blocks_true_nodes;
        client_metrics.metric2_block_subspace_errors = block_subspace_errors;
        client_metrics.metric2_block_details = block_details;
        client_metrics.metric2_mean_block_subspace_error = mean(block_subspace_errors);

        client_metrics.Qhat_aligned_local = Qhat_aligned_local;
        client_metrics.row_block_id_local = row_block_id_local;
        client_metrics.QhatG = Mhat;
        client_metrics.true_support_allowed = support_true_allowed;
        client_metrics.estimated_support = support_est;
        client_metrics.metric3_leakage_error = leakage;
        client_metrics.metric3_support_mismatch = mismatch;
        client_metrics.epsilon = epsilon;

        metrics.clients{k} = client_metrics;
    end
end

%% ========================================================================
function validate_evaluation_inputs(data)

    required_top = {'input', 'clients', 'global_estimated'};
    for i = 1:numel(required_top)
        if ~isfield(data, required_top{i})
            error('data.%s is missing.', required_top{i});
        end
    end

    if ~isfield(data.input, 'A')
        error('data.input.A is missing.');
    end

    if ~isfield(data.global_estimated, 'global_ancestor_matrix')
        error('data.global_estimated.global_ancestor_matrix is missing.');
    end

    K = numel(data.clients);
    for k = 1:K
        client = data.clients{k};

        if ~isfield(client, 'Ok')
            error('data.clients{%d}.Ok is missing.', k);
        end
        if ~isfield(client, 'G')
            error('data.clients{%d}.G is missing.', k);
        end
        if ~isfield(client, 'block_info')
            error('data.clients{%d}.block_info is missing.', k);
        end
        if ~isfield(client, 'block_rq') || ~isfield(client.block_rq, 'Qk')
            error('data.clients{%d}.block_rq.Qk is missing.', k);
        end
        if ~isfield(client, 'estimated_model')
            error('data.clients{%d}.estimated_model is missing.', k);
        end
        if ~isfield(client.estimated_model, 'blocks') || ~isfield(client.estimated_model, 'Qhat')
            error('data.clients{%d}.estimated_model.blocks or Qhat is missing.', k);
        end
    end
end

%% ========================================================================
function label_to_true = get_label_to_true_mapping(data, p)
% Returns a vector label_to_true of length p:
%   label_to_true(j) = true Z node corresponding to intervention label j

    label_to_true = 1:p;  % fallback identity

    if isfield(data, 'global') && isfield(data.global, 'target_node')
        tn = data.global.target_node(:)';
        % target_node is stored as:
        %   target_node(1)=0 for observational
        %   target_node(j+1)=true node for intervention label j
        if numel(tn) >= p+1
            label_to_true = tn(2:p+1);
            return;
        end
    end
end

%% ========================================================================
function Anc = true_ancestor_matrix_from_A(A)
% Build true ancestor matrix from A under convention:
%   Anc(child, ancestor)=1

    P = double(abs(A) > 0);   % P(child,parent)=1
    Anc = transitive_closure_child_ancestor(P);
end

%% ========================================================================
function closure = transitive_closure_child_ancestor(A)
% Convention:
%   A(child,ancestor)=1

    n = size(A, 1);
    closure = double(A ~= 0);

    for mid = 1:n
        for child = 1:n
            if closure(child, mid) == 1
                for anc = 1:n
                    if closure(mid, anc) == 1
                        closure(child, anc) = 1;
                    end
                end
            end
        end
    end

    closure(1:n+1:end) = 0;
end

%% ========================================================================
function M_true = align_square_matrix_label_to_true(M_label, label_to_true, p)
% Align a square matrix indexed by intervention labels to true Z numbering.
%
% Convention preserved:
%   M_label(child_label, anc_label) -> M_true(child_true, anc_true)

    M_true = zeros(p, p);

    nlab = min(size(M_label,1), numel(label_to_true));
    for child_label = 1:nlab
        child_true = label_to_true(child_label);
        for anc_label = 1:nlab
            anc_true = label_to_true(anc_label);
            M_true(child_true, anc_true) = M_label(child_label, anc_label);
        end
    end

    M_true = double(M_true ~= 0);
    M_true(1:p+1:end) = 0;
end

%% ========================================================================
function err = offdiag_hamming_error(A, B)
% Normalized off-diagonal Hamming error

    n = size(A,1);
    mask = ~eye(n);
    err = sum(abs(A(mask) - B(mask))) / sum(mask(:));
end

%% ========================================================================
function [prec, rec, f1] = binary_prf1_offdiag(Atrue, Aest)
% Precision / Recall / F1 ignoring diagonal

    n = size(Atrue,1);
    mask = ~eye(n);

    ytrue = Atrue(mask) ~= 0;
    yest = Aest(mask) ~= 0;

    tp = sum(ytrue & yest);
    fp = sum(~ytrue & yest);
    fn = sum(ytrue & ~yest);

    prec = tp / max(tp + fp, 1);
    rec  = tp / max(tp + fn, 1);

    if prec + rec == 0
        f1 = 0;
    else
        f1 = 2 * prec * rec / (prec + rec);
    end
end

%% ========================================================================
function [Qhat_aligned_local, row_block_id_local, est_blocks_true_nodes] = ...
    align_estimated_Qhat_to_true_local(est, Ok, label_to_true)
% Align estimated Qhat rows (currently indexed by intervention labels / estimated block order)
% to the client's TRUE local node order Ok.
%
% OUTPUT
%   Qhat_aligned_local : pk x nk, rows reordered to match Ok order
%   row_block_id_local : pk x 1, row_block_id_local(i)=estimated block index of node Ok(i)
%   est_blocks_true_nodes : cell array, block nodes in TRUE numbering

    Qhat = est.Qhat;
    blocks = est.blocks;
    pk = numel(Ok);
    nk = size(Qhat, 2);

    Qhat_aligned_local = zeros(pk, nk);
    row_block_id_local = zeros(pk, 1);
    est_blocks_true_nodes = cell(numel(blocks), 1);

    row_start = 1;
    for b = 1:numel(blocks)
        labels_b = blocks{b};
        s = numel(labels_b);
        rows_b = row_start:(row_start + s - 1);
        row_start = row_start + s;

        true_nodes_b = map_labels_to_true_nodes(labels_b, label_to_true);
        est_blocks_true_nodes{b} = true_nodes_b;

        local_idx = nodes_to_local_indices(true_nodes_b, Ok);

        % Put estimated rows into TRUE local order positions.
        % Within-block basis is only identifiable up to subspace equivalence,
        % but this row placement is consistent with the label alignment provided by user.
        Qhat_aligned_local(local_idx, :) = Qhat(rows_b, :);
        row_block_id_local(local_idx) = b;
    end
end

%% ========================================================================
function true_nodes = map_labels_to_true_nodes(labels, label_to_true)
    true_nodes = zeros(size(labels));
    for i = 1:numel(labels)
        true_nodes(i) = label_to_true(labels(i));
    end
end

%% ========================================================================
function local_idx = nodes_to_local_indices(nodes, Ok)
% Convert TRUE node labels into local row indices in client ordering Ok.

    local_idx = zeros(size(nodes));
    for i = 1:numel(nodes)
        pos = find(Ok == nodes(i), 1);
        if isempty(pos)
            error('A mapped true node is not contained in client.Ok.');
        end
        local_idx(i) = pos;
    end
end

%% ========================================================================
function rows = block_rows_from_blocks(blocks, b)
% Return row indices in Qhat corresponding to block b, given stacked block order.

    rows = [];
    offset = 0;
    for i = 1:numel(blocks)
        s = numel(blocks{i});
        if i == b
            rows = offset + (1:s);
            return;
        end
        offset = offset + s;
    end
end

%% ========================================================================
function err = subspace_projection_error(Qhat_block, Qtrue_block)
% Compare row spaces using projection-matrix distance:
%   err = (1/sqrt(2)) * ||P_hat - P_true||_F

    Bhat = orthonormal_row_basis(Qhat_block);
    Btrue = orthonormal_row_basis(Qtrue_block);

    Phat = Bhat' * Bhat;
    Ptrue = Btrue' * Btrue;

    err = norm(Phat - Ptrue, 'fro') / sqrt(2);
end

%% ========================================================================
function B = orthonormal_row_basis(X)
% Return an orthonormal ROW basis of row(X).
% Output B has size r x n and satisfies B*B' = I_r.

    [~, S, V] = svd(X, 'econ');
    s = diag(S);

    tol = max(size(X)) * eps(max(s) + 1);
    r = sum(s > tol);

    if r == 0
        B = zeros(0, size(X,2));
    else
        B = V(:, 1:r)';
    end
end

%% ========================================================================
function allowed = true_support_mask_from_block_structure(block_info)
% Build TRUE support mask under the rule:
%   (i,j) is allowed to be nonzero iff the block of node j is
%   the same as, or an ancestor block of, the block of node i.
%
% IMPORTANT:
%   block_info.block_ancestor_matrix from generated data uses the convention:
%       block_ancestor(anc, desc)=1
%   i.e., ROW = ancestor block, COL = descendant block.

    blocks_local = block_info.blocks_local;
    block_ancestor = block_info.block_ancestor_matrix;  % anc, desc
    num_blocks = numel(blocks_local);

    pk = 0;
    for b = 1:num_blocks
        pk = max(pk, max(blocks_local{b}));
    end

    node_to_block = zeros(pk, 1);
    for b = 1:num_blocks
        node_to_block(blocks_local{b}) = b;
    end

    allowed = zeros(pk, pk);

    for i = 1:pk
        bi = node_to_block(i);  % row node's block (descendant block)
        for j = 1:pk
            bj = node_to_block(j);  % column node's block (candidate ancestor block)

            if bj == bi || block_ancestor(bj, bi) == 1
                allowed(i, j) = 1;
            end
        end
    end
end

%% ========================================================================
function leak = structural_leakage_rate(support_est, support_allowed)
% Fraction of theoretically-forbidden entries that are estimated as nonzero.

    forbidden = (support_allowed == 0);
    denom = sum(forbidden(:));

    if denom == 0
        leak = 0;
    else
        leak = sum((support_est == 1) & forbidden, 'all') / denom;
    end
end

%% ========================================================================
function mm = support_mismatch_rate(support_est, support_allowed)
% Overall support mismatch rate over all entries.

    mm = sum(abs(double(support_est) - double(support_allowed)), 'all') / numel(support_est);
end