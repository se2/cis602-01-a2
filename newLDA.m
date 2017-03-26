function [eigvector, eigvalue] = newLDA(trainLabel, trainFea)

    % Normalizing Data
    trainFea = NormalizeFea(trainFea);

    % Getting Unique lables out of training label column
    uniqueClassLabels = unique(trainLabel);

    % for calculating SW scatter within class
    Sw = 0;
    
    % for calculating SB  scatter between classes
    Sb = 0;
    
    % total mean forcalcluating meean of all data for SB calculation
    totalDataMean = mean(trainFea);

    % calculating scatter within classes
    for i=1:length(uniqueClassLabels)
        
        % getting index of rows of one class
        indices = find(trainLabel == uniqueClassLabels(i));

        % calculating mean of each class
        m = mean(trainFea(indices,:), 1);
        Sw = Sw + (trainFea(indices,:) - m)' * (trainFea(indices,:) - m);
        ni = length(indices);
        Sb = Sb + ni * (m - totalDataMean)' * (m - totalDataMean);
        
    end

    [eigvector, eigvalue] = eig(Sb, Sw);
end
