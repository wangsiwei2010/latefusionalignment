function [H_normalized]= mylocalkernelkmeans(K,A0,cluster_count)

num = size(K,1);
opt.disp = 0;
% K0 = zeros(num);
% for i =1:num
%     Ki = zeros(num);
%     Ki(A(:,i),A(:,i)) = K(A(:,i),A(:,i));
%     K0 = K0 + Ki;
% end
K0 = (1/num)*(A0.*K);
K0= (K0+K0')/2;
[H,~] = eigs(K0,cluster_count,'LA',opt);
% H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);
H_normalized = H;