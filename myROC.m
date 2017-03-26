function [t,f] = myROC(data, pairlabel)

%     convert data to double - use with LBP only
    data = im2double(data);


%     normalize data
    data = NormalizeFea(data);
    
%     call PCA to reduce dimension

    options = [];
    options.ReducedDim=100;
    [eigvectorPCA, eigvaluePCA] = PCA(data, options);
    data = data * eigvectorPCA;
    

%     get number of pairs

    [nPair, nFea] = size(pairlabel);
    Score = [];
        
%     assign label for data
    Label = [];
    for i=1:10
        for j=1:600 * i
            right = 600 * i;
            left = (i-1)*600;
            mid = (left+right)/2;
            if j<=mid && j>left
                Label(:,j) = 1;
            end
            if j>mid && j <= right
                Label(:,j) = 0;
            end
        end
    end
    
%     extract and compute similarity of each pair 
    for i=1:nPair
        indexf1 = pairlabel(i,1);
        indexf2 = pairlabel(i,2);
        f1 = data(indexf1,:);
        f2 = data(indexf2,:);
        subscore = pdist([f1;f2],'cosine');
        Score(:,i) = subscore;
    end

%     reutunr true acceptance and false acceptance rate
    [t,f,thres] = roc(Label, Score);
end
