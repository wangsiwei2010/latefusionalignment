function [ Z ] = l2p( assignments )
%L2P �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[n,~] = size(assignments);
k = length(unique(assignments));
Z = zeros(n,k)

for i=1:n
    index = assignments(i,1);
    Z(i,index) = 1;
end


end

