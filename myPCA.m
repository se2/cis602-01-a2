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

eigvalue = diag(eigvalue);
        
[dump, index] = sort(-eigvalue);
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

maxEigValue = max(abs(eigvalue));
eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
eigvalue(eigIdx) = [];
eigvector(:,eigIdx) = [];

% compute egivector
% eigvector = eigvector(:,1:reduceddim)