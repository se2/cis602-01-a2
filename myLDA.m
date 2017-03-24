function [eigvector, eigvalue] = myLDA(trainLabel, trainFea)

    % call myPCA to reduce dimension
    [eigvectorPCA, eigvaluePCA] = myPCA(trainFea);
    trainFea = trainFea * eigvectorPCA;

    % normalize data
    trainFea = NormalizeFea(trainFea);

    classLabel = unique(trainLabel);
    nClass = length(classLabel);

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

end

