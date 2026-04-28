function data = estimate_global_ancestor_from_clients(data)
%ESTIMATE_GLOBAL_ANCESTOR_FROM_CLIENTS
% Recover global ancestor relation from client-wise estimated block parent relations.
%
% Convention used in THIS file:
%   Matrix entry (i,j)=1 means label j is an ancestor / parent of label i.
%   So: row = descendant, column = ancestor.
%
% Rule:
%   If in some client k, {j} and {i} are both singleton blocks, and
%   {j} is a block-level parent of {i}, then declare j is a global ancestor of i.
%   After collecting all such pairs from all clients, take transitive closure.
%
% INPUT
%   data : struct containing data.clients{k}.estimated_model
%
% OUTPUT
%   data.global_estimated with fields:
%       .direct_parent_evidence      p x p binary matrix
%       .global_ancestor_matrix      p x p binary matrix (transitive closure)
%       .supporting_clients          p x p cell, each entry lists supporting clients
%       .singleton_block_pairs       struct array of local evidence
%
% Convention reminder:
%   (i,j)=1 means j is an ancestor/parent of i.

    if ~isstruct(data) || ~isfield(data, 'clients')
        error('Input must be a data struct with field data.clients.');
    end

    K = numel(data.clients);

    % infer global label universe
    p = infer_global_num_labels(data);

    direct_parent_evidence = zeros(p, p);
    supporting_clients = cell(p, p);

    evidence_records = struct( ...
        'client', {}, ...
        'descendant_label', {}, ...
        'ancestor_label', {}, ...
        'descendant_block_index', {}, ...
        'ancestor_block_index', {});

    rec_count = 0;

    for k = 1:K
        client = data.clients{k};

        if ~isfield(client, 'estimated_model') || isempty(client.estimated_model)
            error('data.clients{%d}.estimated_model is missing. Run estimate_blocks_and_Qk first.', k);
        end

        est = client.estimated_model;

        if ~isfield(est, 'blocks') || ~isfield(est, 'ancestor_matrix')
            error('data.clients{%d}.estimated_model must contain fields blocks and ancestor_matrix.', k);
        end

        blocks = est.blocks;
        ancestor_matrix = est.ancestor_matrix;

        % We need block-level parent relation, not just ancestor relation.
        % Under the convention:
        %   parent_matrix(child, parent)=1 means "parent" is a block-level parent of "child"
        parent_matrix = ancestor_to_parent_matrix(ancestor_matrix);

        num_blocks = numel(blocks);

        % collect singleton blocks
        singleton_block_idx = [];
        singleton_labels = [];

        for b = 1:num_blocks
            if numel(blocks{b}) == 1
                singleton_block_idx(end+1) = b; %#ok<AGROW>
                singleton_labels(end+1) = blocks{b}(1); %#ok<AGROW>
            end
        end

        % For every singleton child <- singleton parent block relation,
        % add global evidence: ancestor label j is parent of descendant label i
        % so matrix entry is (i,j)=1.
        for ii = 1:numel(singleton_block_idx)
            bc = singleton_block_idx(ii);
            i = singleton_labels(ii); % descendant label

            for jj = 1:numel(singleton_block_idx)
                bp = singleton_block_idx(jj);
                j = singleton_labels(jj); % ancestor label

                if bp == bc
                    continue;
                end

                if parent_matrix(bc, bp) == 1
                    % Record the parent-child relation (i,j) for direct parent evidence
                    direct_parent_evidence(i, j) = 1;
                    supporting_clients{i, j} = unique([supporting_clients{i, j}, k]);

                    rec_count = rec_count + 1;
                    evidence_records(rec_count).client = k;
                    evidence_records(rec_count).descendant_label = i;
                    evidence_records(rec_count).ancestor_label = j;
                    evidence_records(rec_count).descendant_block_index = bc;
                    evidence_records(rec_count).ancestor_block_index = bp;
                end
            end
        end
    end

    global_ancestor_matrix = transitive_closure_binary(direct_parent_evidence);

    % No forced upper triangle anymore, just compute the transitive closure correctly
    out = struct();
    out.direct_parent_evidence = direct_parent_evidence;
    out.global_ancestor_matrix = global_ancestor_matrix;
    out.supporting_clients = supporting_clients;
    out.singleton_block_pairs = evidence_records;
    out.convention = '(i,j)=1 means j is an ancestor/parent of i';

    data.global_estimated = out;

end

%% ========================================================================
function p = infer_global_num_labels(data)
% Infer the global maximum intervention label from all clients' estimated blocks.

    p = 0;
    K = numel(data.clients);

    for k = 1:K
        client = data.clients{k};

        if isfield(client, 'estimated_model') && isfield(client.estimated_model, 'blocks')
            blocks = client.estimated_model.blocks;
            for b = 1:numel(blocks)
                if ~isempty(blocks{b})
                    p = max(p, max(blocks{b}));
                end
            end
        end

        % fallback to valid_j if needed
        if isfield(client, 'valid_j')
            vj = client.valid_j;
            vj = vj(vj > 0);
            if ~isempty(vj)
                p = max(p, max(vj));
            end
        end
    end

    if p == 0
        error('Could not infer the global number of intervention labels.');
    end
end

%% ========================================================================
function parent_matrix = ancestor_to_parent_matrix(ancestor_matrix)
% Convert ancestor matrix to parent matrix (Hasse cover relation).
%
% Convention:
%   ancestor_matrix(child, ancestor)=1 means "ancestor" is an ancestor of "child".
%   parent_matrix(child, parent)=1 means "parent" is a direct block parent of "child".

    n = size(ancestor_matrix, 1);
    parent_matrix = zeros(n);

    for child = 1:n
        for parent = 1:n
            if child == parent || ancestor_matrix(child, parent) == 0
                continue;
            end

            has_intermediate = false;
            for mid = 1:n
                if mid == child || mid == parent
                    continue;
                end

                % parent -> mid -> child
                % under convention: ancestor_matrix(child,parent)=1 means parent ancestor of child
                if ancestor_matrix(child, mid) == 1 && ancestor_matrix(mid, parent) == 1
                    has_intermediate = true;
                    break;
                end
            end

            if ~has_intermediate
                parent_matrix(child, parent) = 1;
            end
        end
    end
end

%% ========================================================================
function closure = transitive_closure_binary(A)
% Compute transitive closure of a binary adjacency matrix.
%
% Convention:
%   A(i,j)=1 means j is an ancestor of i.
%   Row = descendant, column = ancestor.

    n = size(A,1);
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

    closure(1:n+1:end) = 0;  % remove diagonal
end