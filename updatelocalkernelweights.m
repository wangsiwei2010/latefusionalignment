function [gamma,obj]= updatelocalkernelweights(HE0,ZH,lambda)


nbkernel = size(ZH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = lambda*HE0+2*diag(ZH);
f = zeros(nbkernel,1);
A = [];
b = [];
Aeq = ones(nbkernel,1)';
beq = 1;
lb  = zeros(nbkernel,1);
ub =  ones(nbkernel,1);

[gamma,obj]= quadprog(H,f,A,b,Aeq,beq,lb,ub);
gamma(gamma<1e-6)=0;
gamma = gamma/sum(gamma);