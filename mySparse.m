load('ORL.mat');
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

% compute beta and residual
R = zeros(nS/10,1);
B = zeros(nS/2,1);
bi = 1;
ri = 1;
r = 5;
trainData1 = [];
testSample1 = [];
% for i=1:nS
%     % step size is 10
%     if (rem(i, 10) == 1)
%         trainData = (fea(i:(i+4),:))';
%         testSample = (fea((i+5):(i+9),:))';
%         B(bi:(bi+r-1),:) = lasso(trainData, testSample(:,1), 'lambda', .1);
% %         R(ri,:) = norm(trainData * B(bi:(bi+r-1),:) - testSample(:,1));
%         bi = bi + r;
%         ri = ri + 1;
%     end
% end

for i=1:nS
    % step size is 10
    if (rem(i, 10) == 1)
        temptrainData = (fea(i:(i+4),:))';
        temptestSample = (fea((i+5):(i+9),:))';
        trainData1 = horzcat(trainData1,temptrainData);
        testSample1 = horzcat(testSample1, temptestSample);
%         B(bi:(bi+r-1),:) = lasso(trainData, testSample(:,1), 'lambda', .1);
%         R(ri,:) = norm(trainData * B(bi:(bi+r-1),:) - testSample(:,1));
%         bi = bi + r;
%         ri = ri + 1;
    end
end
label = zeros(200,1);
Aprime = zeros(200,200);
Rprime = zeros(10,1);
% beta A
A = [];
R = zeros(200,1);
for i=1:10
%     compute A for testSample(i)
    for j=1:200
    A = lasso( trainData1, testSample1(:,i), 'lambda', .1);
    R(j,:) = norm( trainData1(:,j) * A(j,:) - testSample1(:,i));   
    end
%     get min residual and index of min residual
    [vMin, iMin] = min(R);
%     round up to be label
    iMin = ceil(iMin);
%     store label for testSample(i)
    Rprime(i,:) = iMin;
end

% min value and index of min value
% [vMin, iMin] = min(R);

