function KHp = myLocalKernel(KH,tau,p)

numker = size(KH,3);
num = size(KH,1);
gamma = ones(numker,1)/numker;
KC = sumKbeta(KH,gamma);
indx_0 = genarateNeighborhood(KC,tau);
KHp = zeros(num);
for i =1:num
    KHp(indx_0(:,i),i) = KH(indx_0(:,i),i,p);
end
KHp = (KHp+KHp')/2;
[U,V] = eig(KHp);
diagV = diag(V);
diagV(diagV<eps)=0;
KHp = U*diag(diagV)*U';