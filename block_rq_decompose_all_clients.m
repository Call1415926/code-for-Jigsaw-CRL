function data = block_rq_decompose_all_clients(data)
%BLOCK_RQ_DECOMPOSE_ALL_CLIENTS
% Perform block-wise RQ decomposition for each client's Hk:
%   Hk = Rk * Qk
%
% The block structure is taken from:
%   data.clients{k}.block_info
%
% The decomposition follows the user's block-recursive construction:
%   - blocks are processed from ancestors to descendants
%   - for current block Cl, use already computed Q rows of ancestor blocks
%   - project H(Cl,:) onto orthogonal complement of row-space of Q(Anc(Cl),:)
%   - normalize each row to obtain Q(Cl,:)
%   - solve R(Cl, Cl union Anc(Cl)) = H(Cl,:) * pinv(Q(Cl union Anc(Cl),:))
%
% INPUT
%   data : output struct from generate_multi_client_crl_data(...)
%
% OUTPUT
%   data : same struct, with new field
%          data.clients{k}.block_rq
%
% MATLAB target: R2022b

    if ~isstruct(data) || ~isfield(data, 'clients')
        error('Input data must be the struct produced by generate_multi_client_crl_data.');
    end

    K = numel(data.clients);

    for k = 1:K
        client = data.clients{k};

        if ~isfield(client, 'H')
            error('data.clients{%d}.H does not exist.', k);
        end
        if ~isfield(client, 'block_info') || isempty(client.block_info)
            error('data.clients{%d}.block_info does not exist or is empty.', k);
        end

        Hk = client.H;
        block_info = client.block_info;

        rq_info = block_rq_single_client(Hk, block_info);

        data.clients{k}.block_rq = rq_info;
    end

end

%% ========================================================================
function rq_info = block_rq_single_client(Hk, block_info)
%BLOCK_RQ_SINGLE_CLIENT
% Block-wise RQ decomposition for one client.

    [pk, nk] = size(Hk);

    if ~isfield(block_info, 'blocks_local') || ~isfield(block_info, 'block_ancestor_matrix')
        error('block_info must contain fields blocks_local and block_ancestor_matrix.');
    end

    blocks_local = block_info.blocks_local;
    block_ancestor = block_info.block_ancestor_matrix;
    num_blocks = numel(blocks_local);

    if size(block_ancestor,1) ~= num_blocks || size(block_ancestor,2) ~= num_blocks
        error('Dimension mismatch between blocks_local and block_ancestor_matrix.');
    end

    % ------------------------------------------------------------
    % 1. Obtain an order from ancestors to descendants
    % ------------------------------------------------------------
    % block_ancestor(i,j)=1 means block i is ancestor of block j
    % We topologically sort the DAG whose edges are i -> j when i ancestor of j.
    proc_order = topo_sort_blocks(block_ancestor);
    % proc_order = [b1, b2, ..., br], from ancestors to descendants

    % ------------------------------------------------------------
    % 2. Initialize
    % ------------------------------------------------------------
    Qk = zeros(pk, nk);
    Rk = zeros(pk, pk);

    block_records = cell(num_blocks, 1);

    % ------------------------------------------------------------
    % 3. Process blocks one by one
    % ------------------------------------------------------------
    for pos = 1:num_blocks
        b = proc_order(pos);           % current block index
        rows_b = blocks_local{b};      % row indices in Hk / Qk / Rk

        % ancestor blocks of current block b
        anc_blocks = find(block_ancestor(:, b) ~= 0).';
        anc_rows = concatenate_block_rows(blocks_local, anc_blocks);

        % current block raw rows
        H_cur = Hk(rows_b, :);

        % projection to orthogonal complement of row-space of Q(anc_rows,:)
        if isempty(anc_rows)
            H_proj = H_cur;
            P_perp = eye(nk);
            W = zeros(0, nk);
        else
            W = Qk(anc_rows, :);    % already computed ancestor Q rows
            % Project to orthogonal complement of row(W)
            % H_proj = H_cur * (I - pinv(W)*W)
            P_perp = eye(nk) - pinv(W) * W;
            H_proj = H_cur * P_perp;
        end

        % normalize each row to obtain Q rows for current block
        Q_cur = normalize_rows(H_proj);

        % place into Qk
        Qk(rows_b, :) = Q_cur;

        % current admissible columns in Rk:
        % current block + its ancestor blocks
        support_blocks = unique([b, anc_blocks], 'stable');
        support_rows = concatenate_block_rows(blocks_local, support_blocks);

        Q_support = Qk(support_rows, :);

        % solve current block rows of Rk
        % R(rows_b, support_rows) = H_cur * pinv(Q_support)
        R_sub = H_cur * pinv(Q_support);
        Rk(rows_b, support_rows) = R_sub;

        % store block-level record
        rec = struct();
        rec.block_index = b;
        rec.block_rows = rows_b;
        rec.ancestor_blocks = anc_blocks;
        rec.ancestor_rows = anc_rows;
        rec.support_blocks_for_R = support_blocks;
        rec.support_rows_for_R = support_rows;
        rec.H_cur = H_cur;
        rec.W_ancestor_Q = W;
        rec.P_perp = P_perp;
        rec.H_proj = H_proj;
        rec.Q_cur = Q_cur;
        rec.R_sub = R_sub;

        block_records{b} = rec;
    end

    % ------------------------------------------------------------
    % 4. Diagnostics
    % ------------------------------------------------------------
    recon = Rk * Qk;
    recon_error_fro = norm(Hk - recon, 'fro');
    recon_error_rel = recon_error_fro / max(norm(Hk, 'fro'), eps);

    row_norms_Q = sqrt(sum(Qk.^2, 2));

    orthogonality_report = build_block_orthogonality_report(Qk, blocks_local, block_ancestor);

    % ------------------------------------------------------------
    % 5. Output
    % ------------------------------------------------------------
    rq_info = struct();
    rq_info.Hk = Hk;
    rq_info.Rk = Rk;
    rq_info.Qk = Qk;

    rq_info.blocks_local = blocks_local;
    rq_info.block_ancestor_matrix = block_ancestor;
    rq_info.processing_order_ancestor_to_descendant = proc_order;

    rq_info.block_records = block_records;

    rq_info.reconstruction = recon;
    rq_info.reconstruction_error_fro = recon_error_fro;
    rq_info.reconstruction_error_relative = recon_error_rel;

    rq_info.Q_row_norms = row_norms_Q;
    rq_info.orthogonality_report = orthogonality_report;
end

%% ========================================================================
function order = topo_sort_blocks(block_ancestor)
%TOPO_SORT_BLOCKS
% block_ancestor(i,j)=1 means i is ancestor of j.
% Return a topological order from ancestors to descendants.

    r = size(block_ancestor, 1);

    indeg = sum(block_ancestor, 1);  % number of ancestors
    indeg = indeg(:).';

    remaining = true(1, r);
    order = zeros(1, r);

    for t = 1:r
        candidates = find(remaining & (indeg == 0));

        if isempty(candidates)
            error('Block ancestor graph contains a cycle or invalid relation.');
        end

        % deterministic choice: smallest index
        b = candidates(1);
        order(t) = b;
        remaining(b) = false;

        % remove outgoing edges b -> children
        children = find(block_ancestor(b, :) ~= 0);
        indeg(children) = indeg(children) - 1;
        indeg(b) = -1;
    end
end

%% ========================================================================
function rows = concatenate_block_rows(blocks_local, block_indices)
%CONCATENATE_BLOCK_ROWS
% Concatenate row indices of several blocks in the given order.

    rows = [];
    for ii = 1:numel(block_indices)
        b = block_indices(ii);
        rows = [rows, blocks_local{b}]; %#ok<AGROW>
    end
end

%% ========================================================================
function Q = normalize_rows(X)
%NORMALIZE_ROWS
% Normalize each row of X to have Euclidean norm 1.
% Assumes rows are nonzero under the user's model condition.

    norms = sqrt(sum(X.^2, 2));
    tol = 1e-12;

    if any(norms < tol)
        error(['Some projected rows are numerically zero; cannot normalize. ', ...
               'Under your model assumption this should not happen.']);
    end

    Q = X ./ norms;
end

%% ========================================================================
function report = build_block_orthogonality_report(Qk, blocks_local, block_ancestor)
%BUILD_BLOCK_ORTHOGONALITY_REPORT
% For each pair (descendant block, ancestor block), report the Frobenius norm
% of cross inner products Q(desc,:) * Q(anc,:)'.

    num_blocks = numel(blocks_local);

    cross_fro = zeros(num_blocks, num_blocks);
    cross_max_abs = zeros(num_blocks, num_blocks);

    for bi = 1:num_blocks
        rows_i = blocks_local{bi};
        Qi = Qk(rows_i, :);

        for bj = 1:num_blocks
            if block_ancestor(bj, bi) ~= 0
                rows_j = blocks_local{bj};
                Qj = Qk(rows_j, :);

                M = Qi * Qj.';
                cross_fro(bi, bj) = norm(M, 'fro');
                cross_max_abs(bi, bj) = max(abs(M(:)));
            end
        end
    end

    report = struct();
    report.cross_innerproduct_fro = cross_fro;
    report.cross_innerproduct_max_abs = cross_max_abs;
end