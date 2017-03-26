% clear
% close
% 
% % each row is a sample
% load('PIE.mat');
% 
% % number of training data per person
% numTrain = 5;
% 
% trainInd = [];
% testInd = [];
% 
% % make train/test index
% for i = 1: n_per
%     trainInd = [trainInd, (i-1)*n_sub+1: (i-1)*n_sub+numTrain];
%     testInd = [testInd, (i-1)*n_sub+numTrain+1: i*n_sub];
% end
% 
% %generate training and testing data
% trainFea = Data(trainInd,:);
% trainLabel = Label(trainInd,:);
% testFea = Data(testInd,:);
% testLabel = Label(testInd,:);
% tic;
% 
% % applying PCA first
% options=[];
% options.ReducedDim=100;
% [eigvectorPCA, eigvaluePCA] = PCA(trainFea, options);
% trainFea = trainFea * eigvectorPCA;
% testFea = testFea * eigvectorPCA;
% 
% % call LDA
% [eigvector, eigvalue] = runLDA(trainFea, trainLabel);
% 
% ldaTrainFea = trainFea * eigvector;
% ldaTestFea = testFea * eigvector;
% ldaTime = toc;
% 
% % call nearest neighbor classifier of matlab
% predictLabel = knnclassify(ldaTestFea, ldaTrainFea, trainLabel);
% 
% acc = sum(predictLabel == testLabel) / length(testLabel);
% 
% fprintf('the reconition accuracy with lda is %f.\n', acc);
% fprintf('the running time is %f.\n', ldaTime);

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
