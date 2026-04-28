function m3 = aggregate_metric3(metrics)
% Mean of client-wise leakage errors

    K = numel(metrics.clients);
    vals = nan(K,1);

    for k = 1:K
        vals(k) = metrics.clients{k}.metric3_leakage_error;
    end

    m3 = mean(vals, 'omitnan');
end
