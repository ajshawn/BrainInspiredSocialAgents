function idx = trial_time_indices(trials, trialLength)
starts = (trials-1) * trialLength + 1;
ends   = trials * trialLength;

idx = arrayfun(@(s,e) s:e, starts, ends, 'UniformOutput', false);
idx = [idx{:}];   % concatenate
end