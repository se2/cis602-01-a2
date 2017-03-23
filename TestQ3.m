load('ORL.mat')

% call myPCA
options = [];
options.Fisherface = 1;
[eigVector, eigValue] = LDA(gnd,options, fea);
fea = fea * eigVector;