function [H_normalized,gamma,obj] = mylocalizedregmultikernelclustering(K,cluster_count,qnorm,HE0,A0,lambda)

num = size(K,1);
nbkernel = size(K,3);
%% initialize kernel weights
gamma = ones(nbkernel,1)/nbkernel;
%% combining the base kernels
KC  = mycombFun(K,gamma.^qnorm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag = 1;
iter = 0;
while flag
    iter = iter + 1;
    %% update H with KC
    fprintf(1, 'running iteration of the proposed algorithm %d...\n', iter);
    H = mylocalkernelkmeans(KC,A0,cluster_count);
   %% updata base kernels
    IH = eye(num) - H*H';
    Kx = (1/num)*(A0.*IH);
    %% update kernel weights
    ZH = callZH(K,Kx);
    obj(iter)  = callocalObj(HE0,ZH,gamma,lambda);
    [gamma]= updatelocalkernelweights(HE0,ZH,lambda);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% KC  = mycombFun(KA,gamma.^qnorm);
    if iter>2 && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 ||iter>10)
        flag =0;
    end
    KC  = mycombFun(K,gamma.^qnorm);
end
H_normalized = H./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);