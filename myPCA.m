function [eigvector, eigvalue] = pca(data)

% size of data
[M, N] = size(data);

% center data
mn = mean(data);
datacenter = bsxfun(@minus, data, mn);

% calculate the covariance matrix
covariance = (1/ (N - 1)) * (datacenter' * datacenter);
% covariance = cov(data);

% calculate the eigenvectors and eigenvalues 
[eigvector, eigvalue] = eig(covariance);
