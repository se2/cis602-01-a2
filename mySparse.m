load('ORL.mat');

% % reduce dim of data

% % call LDA
% options.Fisherface = 1;
% [eigVector, eigValue] = LDA(gnd, options, fea);

% fea = fea * eigVector;
% 
% call PCA
% [eigVector, eigValue] = PCA(fea);
% fea = fea * eigVector;

% applying PCA first
options=[];
options.ReducedDim=100;
[eigvectorPCA, eigvaluePCA] = PCA(fea,options);
fea = fea * eigvectorPCA;

% normalize data
fea = NormalizeFea(fea);


% size of fea
[nS, nFea] = size(fea);

% get train datas and test samples
trainData = [];
testSample = [];
testLabel = [];

for i=1:nS
    % step size is 10
    if (rem(i, 10) == 1)
        temptrainData = (fea(i:(i+4),:))';
        temptestSample = (fea((i+5):(i+9),:))';
        templabel = gnd((i+5):(i+9),:);
        trainData = horzcat(trainData,temptrainData);
        testSample = horzcat(testSample, temptestSample);
        testLabel = vertcat(testLabel, templabel);
    end
end

% normalize data
trainData = NormalizeFea(trainData);
testSample = NormalizeFea(testSample);

% compute beta and residual
% beta B
B = [];
% residual R
R = [];
% label for test samples
label = [];

for i=1:200
%     compute B for testSample(i)
    B = lasso( trainData, testSample(:,i), 'lambda', .015);
    for j=1:200
        R(j,:) = norm( trainData(:,j) * B(j,:) - testSample(:,i));   
    end
%     get min residual and index of min residual
    [vMin, iMin] = min(R);
%     round up to be label
    iMin = ceil(iMin/5);
%     store label for testSample(i)
    label(i,:) = iMin;
end

% compute accuracy
acc = 1 - nnz((label - testLabel))./200;
fprintf('the classification accuracy is %f.\n', acc);
