function [eigvector, eigvalue] = myLDA(trainLabel, trainFea)


%     call myPCA
    [eigvectorPCA, eigvaluePCA] = myPCA(trainFea);
    trainFea = trainFea * eigvectorPCA;

%     normalize data

    trainFea = NormalizeFea(trainFea);

%     extract class label and number of classes
    classLabel = unique(trainLabel);
    nClass = length(classLabel);
    
%     init Sw, Sb and center of data
    Sw = zeros();

    Sb = zeros();
    centerData = mean(trainFea);
    
%     compute Sw and Sb
    for i=1:nClass
        index = find(trainLabel == classLabel(i));
        classMean = mean(trainFea(index,:));
        Sw = Sw + (trainFea(index,:) - classMean)' * (trainFea(index,:) - classMean);
        [ni, nf] = size(trainFea(index,:));
        Sb = Sb + ni * (classMean - centerData)' * (classMean - centerData);
    end

%     return eigvector and eigvalue
    [eigvector, eigvalue] = eig(Sb, Sw);
end

