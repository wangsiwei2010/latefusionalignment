function [Hstar,Wstar,WP,gamma,obj] = averagemultikernelLatefusionclustering(K,k,lambda,tau)

num = size(K, 2); %the number of samples
numker = size(K, 3); %m represents the number of kernels
maxIter = 100; %the number of iterations
%construct r_p,wp
gamma = ones(numker,1)/numker;
HP = zeros(num,k,numker);
WP = zeros(k,k,numker);
opt.disp = 0;
KH = zeros(num,num,numker);
for p=1:numker % m - kernels
    KH(:,:,p) = myLocalKernel(K,tau,p);
    KH(:,:,p) = (KH(:,:,p)+KH(:,:,p)')/2;
    [Hp, ~] = eigs(KH(:,:,p), k, 'la', opt);
    HP(:,:,p) = Hp;
    WP(:,:,p) = eye(k);
end
Wstar = eye(k);

RpHpwp = zeros(num,k); % k - clusters, N - samples
for p=1:numker
    % Rp2Hpwp = Rp2Hpwp + gamma(p)^2*Hp(:,:,p)*wp(:,:,p);
    RpHpwp = RpHpwp + gamma(p)*(HP(:,:,p)*WP(:,:,p));
end

flag = 1;
iter = 0;
while flag
    iter = iter +1;
    %the first step-- optimize H_star, given Rp,wp
    %equivalent to maximize(sum(rp2*tr(H_star'*Hp*wp)))
    %SVD of Hp'*H_star
    UU = RpHpwp*Wstar';
    [Uh,Sh,Vh] = svd(UU,'econ');
    Hstar = Uh*Vh';
%     Hstar = zeros(num,k);
%     for ipk =1:num
%         [val, indx] = max(UU(ipk,:));
%         Hstar(ipk,indx) = 1;
%     end
    RpHpwp2 = zeros(num,k); % k - clusters, N - samples
    for p=1:numker
        % Rp2Hpwp = Rp2Hpwp + gamma(p)^2*Hp(:,:,p)*wp(:,:,p);
        RpHpwp2 = RpHpwp2 + gamma(p)*HP(:,:,p)*WP(:,:,p);
    end
    [Uw,Sw,Vw] = svd(Hstar'*RpHpwp2,'econ');
    Wstar = Uw*Vw';
    %the second step-- optimizing wp, given Rp,H_star equivalent to maximize(sum(2*tr(wp'*Hp'*H_star)))
    %SVD of Hp'*H_star, we will get one wp for each for_loop as follows
    for p=1:numker
        if gamma(p)>1e-4
            TP = HP(:,:,p)'*(Hstar*Wstar - (RpHpwp - gamma(p)*HP(:,:,p)*WP(:,:,p)));
%             WP0 = zeros(k);
%             for ipk =1:k
%                 [val, indx] = max(TP(:,ipk));
%                 e00 = zeros(k,1);
%                 e00(indx) =1;
%                 WP0(:,ipk) = e00;
%             end
%             WP(:,:,p) = WP0;
            [Up,Sp,Vp] = svd(TP,'econ');
            WP(:,:,p) = Up*Vp';
        end  
    end
    gamma = ones(numker,1)/numker;
    
    RpHpwp = zeros(num,k);    
    for p=1:numker
        RpHpwp = RpHpwp + gamma(p)*HP(:,:,p)*WP(:,:,p);
    end
    obj(iter) = trace((Hstar*Wstar-RpHpwp)'*(Hstar*Wstar-RpHpwp));
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter)
        flag =0;
    end
end
