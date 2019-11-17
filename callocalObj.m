function obj = callocalObj(HE0,ZH,gamma0,lambda)

% ZH = callZH(T,K,A0);
obj = (1/2)*gamma0'*(lambda*HE0+2*diag(ZH))*gamma0;
