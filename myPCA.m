function [eigvector, eigvalue] = pca(data)

% center data
mn = mean(data);
datacenter = bsxfun(@minus, data, mn);

%Normalize data
datacenter = NormalizeFea(datacenter);

% calculate the covariance matrix
covariance = (datacenter' * datacenter);
% covariance = cov(data);

% calculate the eigenvectors and eigenvalues 
[eigvector, eigvalue] = eig(covariance);

