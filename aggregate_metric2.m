function m2 = aggregate_metric2(metrics)
% Mean of client-wise mean block subspace errors

    K = numel(metrics.clients);
    vals = nan(K,1);

    for k = 1:K
        vals(k) = metrics.clients{k}.metric2_mean_block_subspace_error;
    end

    m2 = mean(vals, 'omitnan');
end