function data = estimate_blocks_and_Qk(data, tau)
%ESTIMATE_BLOCKS_AND_QK Estimate blocks and Qk subspaces for each client.
% This function includes all the modifications per your convention
% where new blocks are inserted at the FRONT, and ancestor relationships are updated.
%
% INPUT
%   data : struct containing data.clients{k}.interventions
%   tau : threshold for rank approximation in finite-sample case
%
% OUTPUT
%   data with updated fields for estimated blocks and Qk.

    K = numel(data.clients);

    for k = 1:K
        client = data.clients{k};
        interventions = client.interventions;

        % ---- Collect precision matrices ----
        Theta0 = [];
        labels = [];
        for i = 1:length(interventions)
            if interventions{i}.j == 0
                Theta0 = interventions{i}.Hat_Theta_k; % observational matrix
            else
                labels(end+1) = interventions{i}.j; % add non-zero intervention labels
            end
        end

        % Get the number of interventions
        p = length(labels);
        Theta = cell(1,p);

        for i = 1:p
            for t = 1:length(interventions)
                if interventions{t}.j == labels(i)
                    Theta{i} = interventions{t}.Hat_Theta_k;
                end
            end
        end

        % ---- Calculate Δ_j ----
        Delta = cell(1,p);
        for j = 1:p
            Delta{j} = Theta{j} - Theta0;
        end

        % ---- Block Identification ----
        result = block_identification_FS(Delta, labels, tau);

        % ---- Update client model ----
        client.estimated_model = result;

        % ---- Store results in data structure ----
        data.clients{k} = client;
    end
end

%% ========================================================================
function result = block_identification_FS(Delta, labels, tau)
%BLOCK_IDENTIFICATION_FS Identifies block structures and estimates Qk subspaces.
% Uses finite-sample corrections based on rank approximation and ancestor relationships.

    p = length(labels); % Number of labels
    nk = size(Delta{1}, 1); % Dimension of each sample

    remaining = 1:p; % Remaining block candidates
    blocks = {}; % To store identified blocks
    Qhat = []; % Estimated Qk subspace matrix
    ancestor = []; % To store ancestor relationships

    while ~isempty(remaining)
        W = Qhat; % W is the current Qk matrix (starting as empty)

        r = 1; % Start with rank 1 approximation
        found = false;

        while ~found
            if r > numel(remaining)
                disp('Warning:No candidate satisfied the rank-ratio threshold. Force merging all remaining labels into one final block.')
                Cr = remaining;
                found = true;
                break;
            end
            
            comb = nchoosek(remaining, r); % Generate combinations of size r
            best_score = -inf;
            best_set = [];

            % Check each combination of size r
            for i = 1:size(comb, 1)
                C = comb(i, :);
                M = build_M(Delta, C); % Build the corresponding matrix for the selected blocks

                if ~isempty(W)
                    M = project_orthogonal(M, W); % Project to the orthogonal complement of the current Qk space
                end

                s = svd(M); % Perform SVD on the matrix
                score = sum(s(1:r)) / sum(s); % Calculate the score based on the top r singular values

                if score > best_score
                    best_score = score;
                    best_set = C;
                end
            end

            % If the best score is sufficient, select it
            if best_score >= 1 - tau
                found = true;
                Cr = best_set;
            else
                r = r + 1; % Increase rank if no sufficient match found
            end
        end

        % ---- Algorithm 1: Identify Ancestors ----
        [Qnew, Anc] = identify_ancestors_FS(Cr, Delta, blocks, Qhat, tau);

        % Update ancestor matrix: new block goes to the FRONT
        ancestor = prepend_block_ancestor_matrix(ancestor, Anc);

        % Update Qhat: new block goes to the FRONT
        Qhat = [Qnew; Qhat];

        % Update blocks: new block goes to the FRONT
        blocks = [{labels(Cr)}, blocks];

        % Update remaining blocks
        remaining = setdiff(remaining, Cr);
    end

    result.blocks = blocks;
    result.ancestor_matrix = ancestor;
    result.Qhat = Qhat;
end

%% ========================================================================
function [Qnew, Anc] = identify_ancestors_FS(Cr, Delta, blocks, Qhat, tau)
%IDENTIFY_ANCESTORS_FS Identify ancestor blocks for a given candidate block Cr.
% This function implements the ancestor identification process for finite samples.

    num_blocks = length(blocks);
    Anc = 1:num_blocks; % Initialize ancestors to be all blocks

    M = build_M(Delta, Cr); % Build matrix for the candidate block Cr

    % For each existing block, check if it's a parent of the current candidate
    for b = 1:num_blocks
        W = remove_block(Qhat, blocks, b); % Remove the current block from Qhat

        Mproj = project_orthogonal(M, W); % Project to the orthogonal complement of W

        s = svd(Mproj);
        r = length(Cr); % Rank of the candidate block
        score = sum(s(1:r)) / sum(s); % Rank score

        if score >= 1 - tau
            Anc = setdiff(Anc, b); % If the score is good, consider this block as an ancestor
        end
    end

    % Rebuild the Qk subspace for the identified ancestor blocks
    W = Qhat(rows_of_blocks(blocks, Anc), :);
    Mproj = project_orthogonal(M, W);
    [~, ~, V] = svd(Mproj, 'econ'); % Perform SVD on the projected matrix

    Qnew = V(:, 1:length(Cr))'; % Select the top singular vectors and transpose

    Qnew = normalize_rows(Qnew); % Normalize the rows of Qnew

end

%% ========================================================================
function M = build_M(Delta, C)
%BUILD_M Constructs the matrix for the selected candidate block C from the difference matrices Delta.
    M = [];

    for j = C
        M = [M; Delta{j}];
    end
end

%% ========================================================================
function Mproj = project_orthogonal(M, W)
%PROJECT_ORTHOGONAL Projects matrix M onto the orthogonal complement of matrix W.

    if isempty(W)
        Mproj = M;
    else
        P = eye(size(W, 2)) - pinv(W) * W; % Projection matrix
        Mproj = M * P; % Apply projection
    end
end

%% ========================================================================
function X = normalize_rows(X)
%NORMALIZE_ROWS Normalize each row of matrix X to have unit norm.
    for i = 1:size(X, 1)
        X(i, :) = X(i, :) / norm(X(i, :)); % Normalize each row
    end
end

%% ========================================================================
function rows = rows_of_blocks(blocks, idx)
%ROWS_OF_BLOCKS Get the rows corresponding to the block indices in idx.

    rows = [];
    offset = 0;

    for i = 1:length(blocks)
        s = length(blocks{i});
        if ismember(i, idx)
            rows = [rows offset + (1:s)]; %#ok<AGROW>
        end
        offset = offset + s;
    end
end

%% ========================================================================
function W = remove_block(Qhat, blocks, b)
%REMOVE_BLOCK Removes the rows of block b from Qhat.

    rows = rows_of_blocks(blocks, b);
    W = Qhat;
    W(rows, :) = []; % Remove the corresponding rows
end

%% ========================================================================
function ancestor_new = prepend_block_ancestor_matrix(ancestor_old, Anc_old_idx)
%PREPEND_BLOCK_ANCESTOR_MATRIX
% Insert a new block at the FRONT of the ancestor matrix.
%
% Convention:
%   ancestor_matrix(child, ancestor) = 1
%   i.e., column block is an ancestor of row block.
%
% INPUT
%   ancestor_old : n_old x n_old binary matrix
%   Anc_old_idx  : indices (under OLD indexing) of blocks that are ancestors
%                  of the new block
%
% OUTPUT
%   ancestor_new : (n_old+1) x (n_old+1) binary matrix, where the new block
%                  is block 1, and old blocks are shifted to 2:end
%
% This function also enforces:
%   ancestors of ancestors are also ancestors of the new block.

    n_old = size(ancestor_old, 1);
    ancestor_new = zeros(n_old + 1, n_old + 1);

    % old matrix shifts to bottom-right
    ancestor_new(2:end, 2:end) = ancestor_old;

    if isempty(Anc_old_idx)
        return;
    end

    % New block's ancestor row under OLD indexing
    new_row_old_indexing = zeros(1, n_old);

    for a = Anc_old_idx
        % a itself is an ancestor
        new_row_old_indexing(a) = 1;

        % ancestors of a are also ancestors of the new block
        new_row_old_indexing = double(new_row_old_indexing | ancestor_old(a, :));
    end

    % place into new matrix:
    % new block is row 1, old blocks shift to columns 2:end
    ancestor_new(1, 2:end) = new_row_old_indexing;
end