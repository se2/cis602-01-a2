function [t , f] = myROC(data, pairlabel)

    % convert data to double - use with LBP only
    data = im2double(data);

    % call myPCA to reduce dimension
    [eigvectorPCA, eigvaluePCA] = myPCA(data);
    data = data * eigvectorPCA';

    Score = zeros(1,6000);
    for i=1:6000
        indexf1 = pairlabel(i,1);
        indexf2 = pairlabel(i,2);
        f1 = data(indexf1,:);
        f2 = data(indexf2,:);
        subscore = pdist([f2;f1],'cosine');
        Score(:,i) = subscore;

    end
    Label = zeros(1, 6000);
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

    [t,f,thres] = roc(Label, Score);
    % plotroc(Label, Score);
    fprintf('Done %f.\n');
end

