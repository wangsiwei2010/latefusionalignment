function ZH = callZH(K,tmp)

nbkernel = size(K,3);
% num = size(K,1);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IH = eye(num) - H*H';
% tmp = zeros(num);
% for i =1:num
%     A = zeros(num);
%     A(NS(:,i),NS(:,i)) = IH(NS(:,i),NS(:,i));
%     tmp = tmp + (1/num)*A;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ZH = zeros(nbkernel,1);
for p = 1 : nbkernel
    ZH(p) = trace(K(:,:,p)*tmp);
end