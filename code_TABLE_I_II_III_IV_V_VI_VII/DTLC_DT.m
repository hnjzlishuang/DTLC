function [Z,A] = DTLC_DT(Xs,Xt,Ys,Yt0,options)
if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'beta')
    options.beta = 1.0;
end
if ~isfield(options,'ker')
    options.ker = 'linear';
end
if ~isfield(options,'alpha')
    options.alpha = 1;
end
if ~isfield(options,'non')
    options.non = 1;
end
if ~isfield(options,'eta')
    options.eta = 1;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
alpha = options.alpha;
beta = options.beta;
ker = options.ker;
data = options.data;
non = options.non;
fprintf('DTLC_DT:  data=%s  k=%d  alpha=%f beta=%f\n',data,k,alpha,beta);

%% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

%% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
W = e*e'*C;
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        W = W + e*e';
    end
end
W = W/norm(W,'fro');

%% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

%% Construct inter-class nearest sample and  intra-class farest sample matrix
if ~isempty(Yt0) 
    Ms=dist(Xs,Ys,non);
    Mt=dist(Xt,Yt0,non);
    M=blkdiag(Ms,Mt);
    M = M/norm(M,'fro');
else
    M=zeros(n,n);
end


if strcmp(ker,'primal')
    [A,~] = eigs(X*(W+alpha*M)*X'+beta*eye(m),X*H*X',k,'SM');
    Z = A'*X;
elseif strcmp(ker,'linear')
    K = X'*X;
    [A,~] = eigs(K*(W+alpha*M)*K'+beta*eye(n),K*H*K',k,'SM');
    Z = A'*K;
end

