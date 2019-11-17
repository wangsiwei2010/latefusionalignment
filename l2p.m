function [ Z ] = l2p( assignments )
%L2P 此处显示有关此函数的摘要
%   此处显示详细说明
[n,~] = size(assignments);
k = length(unique(assignments));
Z = zeros(n,k)

for i=1:n
    index = assignments(i,1);
    Z(i,index) = 1;
end


end

