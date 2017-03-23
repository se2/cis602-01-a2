function [eigvector, eigvalue] = lda(trainLabel, trainFea)

[nSmp, nFea] = size(trainFea);

classLabel = unique(trainLabel);
nClass = length(classLabel);
Dim = nClass - 1;

sampleMean = mean(trainFea, 1);
trainFea = (trainFea - repmat(sampleMean, nSmp, 1));

Sw = zeros(900, 900);

for i=1:nClass
    index = find(trainLabel == classLabel(i));
    classMean = mean(trainFea(index,:), 1);
    Sw = Sw + (trainFea(index,:) - classMean)' * (trainFea(index,:) - classMean);
end

centerData = mean(trainFea);

Sb = zeros(900, 900);

for i=1:nClass
    index = find(trainLabel == classLabel(i));
    classMean = mean(trainFea(index,:),1);
    [ni, nf] = size(trainFea(index,:));
    Sb = Sb + ni * (classMean - centerData)' * (classMean - centerData);
end

% S = Sb * Sw.^-1;

[eigvector, eigvalue] = eig(Sb, Sw);
% [eigvector, eigvalue] = eig(S);

