load('ORL.mat');

% call LDA
options.Fisherface = 1;
[eigVector, eigValue] = LDA(gnd, options, fea);
fea = fea * eigVector;

% normalize data
fea = NormalizeFea(fea);

% size of fea
[nS, nFea] = size(fea);

% compute beta and residual
R = zeros(nS/10,1);
B = zeros(nS/2,1);
bi = 1;
ri = 1;
r = 5;
for i=1:nS
    % step size is 10
    if (rem(i, 10) == 1)
        trainData = (fea(i:(i+4),:))';
        testSample = (fea((i+5):(i+9),:))';
        B(bi:(bi+r-1),:) = lasso(trainData, testSample(:,1), 'lambda', .1);
        R(ri,:) = norm(trainData * B(bi:(bi+r-1),:) - testSample(:,1));
        bi = bi + r;
        ri = ri + 1;
    end
end

% min value and index of min value
[vMin, iMin] = min(R);

