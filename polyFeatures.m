function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


X_poly = zeros(numel(X), p);

X_poly=X;
T=X;
for i=2:p
 T=X.^i;
 X_poly=[X_poly T];
end

  

% =========================================================================

end