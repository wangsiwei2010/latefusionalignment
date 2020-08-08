clear
clc
warning off;

path = '.\';
addpath(genpath(path));

%%addpath('C:\Program Files\Mosek\8\toolbox\r2014a');

dataName = 'ccv'; %%% flower17; flower102; proteinFold,caltech101_mit,UCI_DIGIT,ccv
%% %% washington; wisconsin; texas; cornell
%% caltech101_nTrain5_48
%% proteinFold
load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
% load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KH = kcenter(KH);
KH = knorm(KH);
K0 = zeros(num,num);

qnorm = 2;

opt.disp = 0;

for p=1:numker % m - kernels
    KH(:,:,p) = (KH(:,:,p)+KH(:,:,p)')/2;
    [Hp, ~] = eigs(KH(:,:,p), numclass, 'la', opt);
    K0 = K0 + (1/numker)*KH(:,:,p);
    HP(:,:,p) = Hp;
end

[H0,~] = eigs(K0, k, 'la', opt);

    
%%---The Proposed1---%%
tic
lambdaset9 = 2.^[-15:1:15];
tauset9 = [0.1:0.1:1];
accval9 = zeros(length(tauset9),1);
nmival9 = zeros(length(tauset9),1);
purval9 = zeros(length(tauset9),1);
time  = zeros(length(lambdaset9),1);


for it =1:length(lambdaset9)
    tic;
    num8 = round(tauset9(10)*num);
    [H_normalized9,WP9,gamma9,obj9] = multikernelLatefusionAlignmentclustering4(HP,numclass,...
        lambdaset9(it),num8,Y,H0);
    res9 = myNMIACC(H_normalized9,Y,numclass);
    accval9(it) = res9(1);
    nmival9(it)= res9(2);
    purval9(it) = res9(3);
    t = toc;
    time(it,1)=t;
end
res(:,9) = [max(max(accval9)); max(max(nmival9));max(max(purval9))];


