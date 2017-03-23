load('LFW_label.mat') 
load('LFW_VGG.mat')

% % convert data to double - use with LBP only
% fea = im2double(fea);

% call myPCA to reduce dimension
[eigvectorPCA, eigvaluePCA] = myPCA(fea);
fea = fea * eigvectorPCA';

Score = zeros(1,6000);
for i=1:6000
    indexf1 = imgIdx(i,1);
    indexf2 = imgIdx(i,2);
    f1 = fea(indexf1,:);
    f2 = fea(indexf2,:);
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
plot(f,t);
