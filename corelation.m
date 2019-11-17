function M = corelation(HP,WP)
%CORELATION �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
m = size(HP,3);
M = zeros(m,m);

for i=1:m
    for j=1:m
        M(i,j) = trace(WP(:,:,i)'*HP(:,:,i)'*HP(:,:,j)*WP(:,:,j));
    end
end
M = (M+M')/2+1e-6*eye(m);


end

