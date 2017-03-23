function [eigvector, eigvalue] = lda(trainLabel, trainFea)

% call myPCA to reduce dimension
[eigvectorPCA, eigvaluePCA] = myPCA(trainFea);
trainFea = trainFea * eigvectorPCA; 

% normalize data
trainFea = NormalizeFea(trainFea);

[nSmp, nFea] = size(trainFea);

classLabel = unique(trainLabel);
nClass = length(classLabel);

sampleMean = mean(trainFea, 1);
trainFea = (trainFea - repmat(sampleMean, nSmp, 1));

Sw = zeros();

for i=1:nClass
    index = find(trainLabel == classLabel(i));
    classMean = mean(trainFea(index,:), 1);
    Sw = Sw + (trainFea(index,:) - classMean)' * (trainFea(index,:) - classMean);
end

centerData = mean(trainFea);

Sb = zeros();

for i=1:nClass
    index = find(trainLabel == classLabel(i));
    classMean = mean(trainFea(index,:),1);
    [ni, nf] = size(trainFea(index,:));
    Sb = Sb + ni * (classMean - centerData)' * (classMean - centerData);
end

[eigvector, eigvalue] = eig(Sb, Sw);


