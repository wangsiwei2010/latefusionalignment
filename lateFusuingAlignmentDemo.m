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

qnorm = 2;

opt.disp = 0;

for p=1:numker % m - kernels
    KH(:,:,p) = (KH(:,:,p)+KH(:,:,p)')/2;
    [Hp, ~] = eigs(KH(:,:,p), numclass, 'la', opt);
    HP(:,:,p) = Hp;
end


    
%%---The Proposed1---%%
tic
lambdaset9 = 2.^[-15:1:15];
lambda2set9 = 2.^[-15:1:15];
tauset9 = [0.1:0.1:1];
accval9 = zeros(length(tauset9),1);
nmival9 = zeros(length(tauset9),1);
purval9 = zeros(length(tauset9),1);
time  = zeros(length(lambdaset9),1);

% for it =1:length(lambdaset9)
%     tic;
%     num8 = round(tauset9(10)*num);
%     [H_normalized9,WP9,gamma9,obj9] = multikernelLatefusionAlignmentclustering2(KH,numclass,...
%         lambdaset9(it),num8,Y);
%     res9 = myNMIACC(H_normalized9,Y,numclass);
%     accval9(it) = res9(1);
%     nmival9(it)= res9(2);
%     purval9(it) = res9(3);
%     t = toc;
%     time(it,1)=t;
% end
% res(:,9) = [max(max(accval9)); max(max(nmival9));max(max(purval9))]; 

% for it =1:length(lambdaset9)
%     tic;
%     for ij=1:length(lambda2set9)       
%         num8 = round(tauset9(10)*num);
%         [H_normalized9,WP9,gamma9,obj9] = multikernelLatefusionAlignmentclustering3(KH,numclass,...
%             lambdaset9(it),num8,Y,lambda2set9(ij));
%         res9 = myNMIACC(H_normalized9,Y,numclass);
%         accval9(it,ij) = res9(1);
%         nmival9(it,ij)= res9(2);
%         purval9(it,ij) = res9(3);
%     t = toc;
%  
%     time(it,ij)=t;
%     end
% end
% res(:,9) = [max(max(accval9)); max(max(nmival9));max(max(purval9))]; 


for it =1:length(lambdaset9)
    tic;
    num8 = round(tauset9(10)*num);
    [H_normalized9,WP9,gamma9,obj9] = multikernelLatefusionAlignmentclustering4(KH,numclass,...
        lambdaset9(it),num8,Y);
    res9 = myNMIACC(H_normalized9,Y,numclass);
    accval9(it) = res9(1);
    nmival9(it)= res9(2);
    purval9(it) = res9(3);
    t = toc;
    time(it,1)=t;
end
res(:,9) = [max(max(accval9)); max(max(nmival9));max(max(purval9))];


% %%---Local Kernel Alignment MKC (IJCAI2016)--%%%%%%%%%%%%
% tauset10 = [0.5:0.05:0.6];
% lambdaset10 = 2.^[-5:2:-3];
% acc10 = zeros(length(tauset10),length(lambdaset10));
% nmi10 = zeros(length(tauset10),length(lambdaset10));
% pur10 = zeros(length(tauset10),length(lambdaset10));
% for it =1:length(tauset10)
%     %%Calculate Neighborhood of each sample
%     NS = genarateNeighborhood(KC3,round(tauset10(it)*num));%% tau*num
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     for il =1:length(lambdaset10)
% 
%         [H_normalized10,gamma10,obj10] = mylocalizedregmultikernelclusteringAdapSampling...
%             (KH,numclass,qnorm,HE0,NS,lambdaset10(il));
%         res10 = myNMIACC(H_normalized10,Y,numclass);
%         acc10(it,il) = res10(1);
%         nmi10(it,il) = res10(2);
%         pur10(it,il) = res10(3);
%     end
% end
% res(:,10) = [max(max(acc10));max(max(nmi10));max(max(pur10))];

%okkc
% tic
% [assignment,~] = okkc(A,numclass,'sip');
% res10 = myNMIACC(assignment,Y,numclass);
% 
% acc10(1) = res10(1);
% nmi10(1) = res10(2);
% pur10(1) = res10(3);
% t = toc;
% time1(1,1)=t;
