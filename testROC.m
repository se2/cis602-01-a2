load('LFW_label.mat')
load('LFW_LBP.mat')
LBP = fea;
load('LFW_VGG.mat')
VGG = fea;

[tLBP, fLBP] = myROC(LBP, imgIdx);
[tVGG, fVGG] = myROC(VGG, imgIdx);

plot(tLBP, fLBP, tVGG, fVGG);
xlabel('False reject rate')
ylabel('False accept rate')
legend('LBP','VGG')