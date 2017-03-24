load('ORL.mat')

% call LDA
options = [];
options.Fisherface = 1;
[eigVector, eigValue] = LDA(gnd, options, fea);
fea = fea * eigVector;

% normalize data
fea = NormalizeFea(fea);

% compute B
trainData = fea(1:5,:);
testSample = fea(6:10,:);
[nS, nFea] = size(trainData);

B = zeros(nS,1);

for i=1:nS
   subB = lasso(trainData(i,:), testSample(i,:));
end