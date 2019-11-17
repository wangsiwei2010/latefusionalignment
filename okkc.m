function [Z,sdi,theta,itr] = okkc(Kmix,K, optmethod)

%function [Z,sdi,theta,itr] = okkc(Kmix,K)
%
% Clustering the data by optimizing the kernels 
%
%INPUTS:
% Kimx    = a struct cell array containing the centered kernels
% K       = the number of clusters
%
%OUTPUTS:
% Z     = the NxK binary cluster assignment matrix
% sdi   = the ordered (from large to small) eigenvalues of the combined kernel matrix
% theta = the coefficients of kernel matrices, the final coefficient corresponds to the inverse of the regularization parameter
% itr   = the number of okkc iterations

%Embedded functions:  [Z,Y,e,e2,e3] = kernel_kmeans(Kernel,K,Max_Its,Zold) :  The kernel K-means clustering algorithm
%                     [theta,B,E,sis,t]=sip_LSSVM_Linf_MKL_multiclass_jointlambda(kernels,A) : The SIP LSSVM formulation for kernel fusion
%                                        with joint lambda estimation
		      
%

%Author:  Shi Yu,      Nov 2009.


%thetaarr =[];
deltaarr =[];
p = size(Kmix,2);
n = size(Kmix{1},1);


Kmean = zeros(n,n);

for loop=1:1:p
 Kmix{loop} = (Kmix{loop} + Kmix{loop}')/2;
 Kmean = Kmean + Kmix{loop};
end
Kmean = Kmean ./ p;

theta = ones(1,p)/p;

Kmean = Kmix{1};


tol = 0.1;
	

	%Kint = Kmix{1};
        [V,D] = eig(Kmean);
	di = diag(D);
	[sdi,ix] = sort(di,'descend');

	V = V(:,ix(1:K));
	%[centers,assignments,error] = km(V',K,20,0.000001);
	assignments = kmeans(V, K);

	Z = l2p(assignments);
	
	
	objold = -9999;

	for itr=1:50
		disp 'iteration';
		itr
		%L = Z*((Z'*Z)^-0.5);

		
		L = ones(n,K);
		L(find(Z==0))=-1;
		
		mu=0;
		B=0;
		E=0;
		sis=0;

		switch optmethod
			case 'sip'
			while (mu==0)
				[mu,B,E,sis]=okkc_sip_LSSVM_Linf_MKL_multiclass_jointlambda(Kmix,L);
			end

			case 'qcqp'
			[obj,B,t,mu,stq] =okkc_qoqc_LSSVM_Linf_multiclass_jointlambda(Kmix,L);

		end

		Koptimal = zeros(size(Kmix{1}));
		for loop=1:1:p
			Koptimal =Koptimal+ Kmix{loop}*mu(loop);
		end
		Koptimal = Koptimal + mu(end)*eye(n, n);
        Koptimal = kcenter(Koptimal);
        Koptimal = knorm(Koptimal);

		[Vnew,D] = eig(Koptimal*Koptimal);
		di = diag(D);
		[sdi,ix] = sort(di,'descend');

		Vnew = Vnew(:,ix(1:K));
		%[centers,assignments,error] = km(V',K,20,0.000001);
		assignments = kmeans(Vnew, K);
		Znew = l2p(assignments);
		thetanew = mu';
		
		deltaV = norm(V-Vnew)/norm(Vnew)

		deltaarr = [deltaarr deltaV];
		
		if (size(deltaarr,2)==2)
			if deltaV<tol || (size(deltaarr,2)==3)
				break;
			end
		end
		
		Z = Znew;
		theta = thetanew;
		V = Vnew;

	end	

%[Z,error] = kernel_kmeans(Koptimal,K,50);
	


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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this is a little utility functino which returns a random integer between 1
%& Max_Int.
function u = rand_int(Max_Int)
u=ceil(Max_Int*rand);




function [theta,B,E,sis]=okkc_sip_LSSVM_Linf_MKL_multiclass_jointlambda(kernels,A)


% SiP for multiple class Linf-norm LSSVM MKL solver
% kernels  a cell object of multiple centered kernels
% A is a nxk matrix of labels, where n is the number of data samples, 
% k is the number of classes, notice that this version requires that A(:,j)^-2 = ones, for all j
%
% Output variables 
% theta:  kernel coefficients
% B:      the dual variables 
% E:      the dummy variable checking the covergence
% sis:    the dummy variable of f(alpha)
% t:      the costed CPU time
%

% Notice:
% The program is for L_inf LS-SVM MKL with joint estimated lambda (regularization parameter) 
% the single LSSVM is solved by the included function linsolve_LSSVM
% the Linf-norm is optimized by the  solve_lp function based on Matlab "linprog" function

% Coded by Shi Yu shi.yu@esat.kuleuven.be and Tillmann Falck tillmann.falck@esat.kuleuven.be, 2009 

time0 = cputime;

N = length(kernels{1});
p = size(A,2);  % size of classes


% add an identity matrix in the kernel fusion
kernels = [kernels {eye(N,N)}];

B = rands(N,p)/2;
sis = compute_sis(kernels, A, B);

E=[];
iter=0;

for n=1:1:10
    iter = iter+1;
    disp(n)
    [theta,gamma] = solve_lp(sis);
    Omega = zeros(size(kernels{1}));
    for n=1:length(kernels)
      Omega = Omega + theta(n) * kernels{n};
    end

    % check the condition of the combined kernel matrix, it can be very singular because of the bad initial guess of B
    cond(Omega)
    if cond(Omega)>1E+20

	disp 'Matrix too singular, regenerating initial matrix';
	theta=0;
	B = 0;
	E = 0;
	sis =0;
	return;
    end
    [B, trash2, trash3] = linsolve_LSSVM(Omega, A);
    S = compute_sis(kernels, A, B);
    sis = [sis; S]
    eps = 1+S*theta/gamma;

    E = [E eps];
    if abs(eps) < 5e-4 || iter == 5
        t = cputime - time0;
        break
    end
end

disp(n)


function sis=compute_sis(kernels, A, B)

N = length(kernels);
p = size(A,2);  % size of classes
sis = zeros(1,N);
c = - sum(sum(B));

for n=1:N-1
	sis(n)= c;
   for j = 1:1:p
	sis(n) = sis(n) + 0.5 * B(:,j)' *diag(A(:,j))* kernels{n} * diag(A(:,j))*B(:,j);
   end
end

sis(N) = c;
for j=1:1:p
  sis(N) = sis(N) + 0.5 * B(:,j)'*kernels{N}*B(:,j);
end

function [theta,gamma]=solve_lp(Sis)
S=size(Sis);

x0= ones(S(2),1)/S(2);
options = optimset('Display','iter','MaxIter',5);
ub=[];

[theta,gamma] = linprog([-1, zeros(1,S(2))], [ones(S(1), 1), -Sis], zeros(S(1),1), [0, ones(1,S(2))], 1, [-inf, zeros(1,S(2))],ub,x0,options);
theta(1) = [];


function  [alpha,beta,b] =linsolve_LSSVM(Omega,L)
% solve LS-SVM as linear problem, the formulation adapted here is different from the formulation given by Suykens et al.
n = size(Omega,1);  % number of data
p = size(L,2);  % number of classes

for loop = 1:1:p
	Y{loop} = diag(L(:,loop));
end
onevec = ones(size(L(:,1)));
H =[0 onevec';onevec Omega];

J = [zeros(1,p); 1./L];
sol = linsolve(H,J);
beta = sol(2:end,:);
b = sol(1,:);

alpha = zeros(size(beta));
for loop=1:1:p
	alpha(:,loop) = inv(Y{loop})*beta(:,loop);
end



function  [obj,B,t,mu,stq] =okkc_qoqc_LSSVM_Linf_multiclass_jointlambda(Kmix,L)

% an yet inefficient solution to solve the Linf-norm LSSVM MKL problem as QCQP in MOSEK
% 
% input variables
% Kmix is the cell array of multiple kernels
% L is the matrix of labels, for multi-class case, the class number is equal to the column number L
% lambda is the reguralization parameter
%
% output variables:
% obj is the optimum of the objective function
% B is the matrix of dual variables
% t is the dummy variable in optimization
% mu are the kernel coefficients
% stq is the costed CPU time



% the bias term can be solved independently after B is obtained

% code written by Shi Yu shi.yu@esat.kuleuven.be
% ESAT, K.U.Leuven, B3001, Heverlee-Leuven, Belgium


% read how many data
numD = size(Kmix{1},1);


dummK = eye(numD,numD);


Kmix = [Kmix dummK];

% read how many kernels
numP = size(Kmix,2);

% read how many clusters
numK = size(L,2);


% creat cluster indicator matrix L from partition matrix P
%L = P*(P'*P)^-0.5;


% init the mosek problem
clear prob;


% c vector .
c = -ones(numD*numK+1,1);
c(end)=1;
prob.c =c;
clear c;




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


obj = 0.5*t - sum(beta);