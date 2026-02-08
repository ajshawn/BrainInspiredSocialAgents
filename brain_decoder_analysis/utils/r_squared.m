function R2 = r_squared(actual, predicted)
%R_SQUARED Computes the coefficient of determination (R²)
% Ensure inputs are column vectors
actual    = actual(:);
predicted = predicted(:);
% Check for equal length
if length(actual) ~= length(predicted)
    error('Input vectors "actual" and "predicted" must be the same length.');
end
% Compute residual sum of squares (SS_res)
SS_res = sum((actual - predicted).^2);
% Compute total sum of squares (SS_tot)
SS_tot = sum((actual - mean(actual)).^2);
% Compute R-squared
R2 = 1 - (SS_res / SS_tot);
end