function [Z,theta,itr] = naml(Kmix,K,lambda)

Kint = zeros(size(Kmix{1}));
for i=1:1:size(Kmix,2)
 Kint = Kint + Kmix{i};
end
Kint = Kint / size(Kmix,2);
%%Kint = center_kernel(Kint);
Kint = kcenter(Kint);


[Z,error] = kernel_kmeans(Kint,K,50);
                                                                                      
tol = 0.01;

olde = -1;
maxe = 0;
convergence = 0;
[N,D]=size(Kint);


L = Z*((Z'*Z)^-0.5);

itr =1;
Lnew = L;

while (~convergence)
      disp '***********************************'; 
      % obtaining Q given L and G
      Sk1= Kint *L*L'*Kint;
      [L Lnew]
      q = rank(Sk1);
      Sk2= Kint*Kint + lambda*Kint;

      pinvSk2 = pinv(Sk2'*Sk2)*Sk2';
      Qsolve = pinvSk2*Sk1;
      [Sq,Dq,Vq] = svd(Qsolve);
      Q = Sq(:,1:q);
      
      % obtaining G given Q and L
      [obj,B,t,w,stq] =qoqc_LSSVM_multiclass_correct(Kmix,L,lambda)
      %[w,B,S,time]=silp5(Kmix,L,lambda);
      %[w,Knew,weights] =qoqc2(Kmix,L,lambda,theta);

	Knew = zeros(size(Kmix{1}));
	for loop=1:1:size(Kmix,2)
		Knew =Knew+ Kmix{loop}*w(loop);
	end
      Kint = Knew;

      Lcond = Q'*(Knew*Knew+lambda*Knew)*Q;

      
%      cond(Knew*Knew+lambda*Knew);
%      if abs(cond(Knew*Knew+lambda*Knew)-1)<tol
%          G = Knew *inv(Knew*Knew+lambda.*Knew)*Knew;
%      else
%          G = Knew *pinv(Knew*Knew+lambda.*Knew)*Knew;
%      end    
%      [Sw,Dw,Vw]=svd(G);
%      Sw = Sw(:,1:K);
      
%      Lnew = Sw;

      %cond(Knew*Knew+lambda*Knew)
      if abs(cond(Lcond)-1)<0.1
          G = Knew*Q*inv(Lcond)*Q'*Knew;
      else
          G = Knew*Q*pinv(Lcond)*Q'*Knew;
      end    
      [Sw,Dw,Vw]=svd(G);

      diag(Dw)
      Sw = Sw(:,1:K);
      
      
      Lnew = Sw;
      
      e = trace(Lnew'*G*Lnew);
      %e = trace(L'*G*L);
      maxe = max(e,maxe);
	
      theta = w;
   	if (abs(olde -e)<tol && abs(maxe-e)<tol)
	   convergence = 1;
            
		 [Q,R]= qr(Lnew');
		 R11 = R(1:K,1:K);
		 R11 = R11 + 0.00001*eye(K,K);
		 Rnew = inv(R11) * R;
		 [rmax,IDX]= max(Rnew);
		 IDX = IDX';
		 Znew = zeros(size(Z));
		 for i=1:1:size(Znew,2)
		    Znew(find(IDX==i),i)=1; 
		 end    
		 Zspec = Znew;	     
		 return;

	      else
		   olde = e;
		  fprintf('Error = %f\n', e);
		  L = Lnew;
		  itr = itr+1;
	  end    
end


function  [obj,B,t,mu,stq] =qoqc_LSSVM_multiclass_correct(Kmix,L,lambda)

% read how many kernels
numP = size(Kmix,2);

% read how many clusters
numK = size(L,2);

% read how many data
numD = size(Kmix{1},1);

% creat cluster indicator matrix L from partition matrix P
%L = P*(P'*P)^-0.5;



%variables
% B_1 B_2 ,..., B_K, t


% init the mosek problem
clear prob;


% c vector .
c = -ones(numD*numK+1,1);
c(end)=0.5;
prob.c =c;
clear c;


% the quadratic term in objective function

prob.qosubi = [];
prob.qosubj = [];
prob.qoval = [];


for kernelLoop = 1:1:numK
	
	prob.qosubi = [prob.qosubi numD*(kernelLoop-1)+1:numD*(kernelLoop-1)+numD];
	prob.qosubj = [prob.qosubj numD*(kernelLoop-1)+1:numD*(kernelLoop-1)+numD];
	prob.qoval =  [prob.qoval 1/lambda*ones(1,numD)];
end
prob.qosubi = [prob.qosubi numK*numD+1]';
prob.qosubj = [prob.qosubj numK*numD+1]';
prob.qoval = [prob.qoval 0]';


% Next quadratic terms in the constraints .
% constraint of 1st kernel
prob.qcsubk =[];
prob.qcsubi =[];
prob.qcsubj = [];
prob.qcval =[];
prob.a = [];
prob.buc = [];
prob.blc=[];
prob.blx = -inf*ones(numD*numK+1,1);  % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)
prob.bux = +inf*ones(numD*numK,1);     % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)
prob.bux = [prob.bux; +inf];  % the dual variables are unconstrained, the dummy variable is also unconstrained (actually >=0)

yacc =[];
for clusterLoop = 1:1:numK
	    yacc= [yacc; L(:,clusterLoop)];
end
yacc =[yacc; 0];


for kernelLoop = 1:1:numP
    Qk=[];

    for clusterLoop = 1:1:numK
	    clusterLoop
	    Y = diag(L(:,clusterLoop));
	    Qk=[Qk;sparse(numD,(clusterLoop-1)*numD) 2*Y*Kmix{kernelLoop}*Y sparse(numD,(numK-clusterLoop)*numD+1)];
    end

    Qk=[Qk; sparse(1,numD*numK+1)];

	trQk = tril(Qk);
	trQk = sparse(trQk);
	[i,j,s] = find(trQk);	
	
	clear Qk;

	prob.qcsubk = [prob.qcsubk; kernelLoop*ones(size(i))];
	prob.qcsubi = [prob.qcsubi i'];
	prob.qcsubj = [prob.qcsubj j'];
	prob.qcval =  [prob.qcval s'];

	prob.a = [prob.a; sparse(1,numK*numD) -1];   %  a'*Y*K*Y*a - 2*gamma <=0
	prob.buc = [prob.buc; 0];
        prob.blc = [prob.blc; -inf];


end


% linear constraints  (Ya)'*\vec 1 = 0

size(yacc')
size(prob.a)
prob.a = [prob.a; yacc'];

prob.buc = [prob.buc; 0];
prob.blc = [prob.blc; 0];

prob.qcsubi = prob.qcsubi';
prob.qcsubj = prob.qcsubj';
prob.qcval = prob.qcval';

size(prob.c)
size(prob.qcsubi)

tic;
[r,res] = mosekopt ('minimize', prob);
stq =toc;

% get the optimal dual variable solution

size(res.sol.itr.xx)

beta = res.sol.itr.xx(1:numK*numD);


B = [];

for loop=1:1:numK
 B = [B beta((loop-1)*numD+1:loop*numD)];
end

t = res.sol.itr.xx(end);

mu = res.sol.itr.suc(1:numP);
mu = mu.*2;

obj = 0.5 * t + 0.5/lambda* (beta'*beta)- sum(beta);

function [Z,Y] = kernel_kmeans(Kernel,K,Max_Its,Zold)

%This is a simple implementation of Kernel K-means clustering - an
%interesting paper which proposed kernel based Kmeans clustering is [1]
%Girolami, M, Mercer Kernel-Based Clustering in Feature Space, 
%IEEE Trans Neural Networks, 13(3),780 - 784, 2002. 
%INPUTS:
% Kernel  = the kernel matrix for clustering
% K       = the number of clusters
% Max_Its = maximal number of iterations
% Zold    = an optional parameter to use the Zold as a starting point
%
%OUTPUTS:
% Z     = the NxK binary cluster assignment matrix
% Y     = the matrix of disimilarities


[N,D]=size(Kernel);


if nargin < 4
	%initialise the indictaor matrix to a random segmentation of the data
	Z = zeros(N,K);
	for n = 1:N
	  Z(n,rand_int(K)) = 1; 
	end
 else
	Z = Zold;
end

for k=1:1:K 
 if (sum(Z(:,k))==0)
     c = rand_int(N);
     Z(c,:) = zeros(1,K);
     Z(c,k)==1;
 end
end

%main loop
for its = 1:Max_Its
    %compute the similarity of each data point to each cluster mean in
    %feature space - note we do not need to compute store or update a mean
    %vector s we are using the kernel-trick - cool eh?

    for k=1:K
        Nk = sum(Z(:,k));
        Y(:,k) = diag(Kernel) - 2*Kernel*Z(:,k)./Nk + Z(:,k)'*Kernel*Z(:,k)./(Nk^2);
    end

    %Now we find the cluster assignment for each point based on the minimum
    %distance of the point from the mean centres in feature space using the
    %Y matrix of dissimilarities
    %for yLoop = 1:N
    %   Y(yLoop,:)=Y(yLoop,:)./max(Y(yLoop,:));
    %end
    %Y
    [i,j]=min(Y,[],2);
    
    %this simply updates the indictor matrix Z refleting the new
    %allocations of data points to clusters
    Z = zeros(N,K);
    for n=1:N
        Z(n,j(n)) = 1;
    end
    
end


%return the clutsers that each data point has been allocated to
for n=1:N
    z(n) = find(Z(n,:));
end

function u = rand_int(Max_Int)
u=ceil(Max_Int*rand);