function [eigvector, eigvalue] = myPCA(data)

    % center data
    mn = mean(data);
    datacenter = bsxfun(@minus, data, mn);

    % normalize data
    datacenter = NormalizeFea(datacenter);

    % calculate the covariance matrix
    covariance = (datacenter' * datacenter);

    % calculate the eigenvectors and eigenvalues 
    [eigvector, eigvalue] = eig(covariance);

end