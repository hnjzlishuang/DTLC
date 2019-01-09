function [Z,A] = DICD(Xs,Xt,Ys,Yt0,options)

% Load algorithm options

if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 0.1;
end
if ~isfield(options,'ker')
    options.ker = 'primal';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
lambda = options.lambda;
ker = options.ker;
gamma = options.gamma;
data = options.data;
beta = options.beta;

fprintf('DICD:  data=%s  k=%d  lambda=%f  beta=%f\n',data,k,lambda,beta);

% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
Xs=X(:,1:ns);
Xt=X(:,ns+1:end);
C = length(unique(Ys));

Yall=[Ys;Yt0];
ratio=options.ratio;

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C;
% M = e*e';
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e';
    end
end
M = M/norm(M,'fro');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% construct distance between each class
BBT1=[];

for c = reshape(unique(Ys),1,C)
    SClassNum(c)=length(find(Ys==c));
    TClassNum(c)=length(find(Yt0==c));
    ClassNum(c)=length(find(Yall==c));
end

BBS1=zeros(ns,ns);
for i=1:ns
    BBS1(i,i) = ns/SClassNum(Ys(i))*(SClassNum(Ys(i)));
    for j=i+1:ns
        if Ys(i)==Ys(j)
            BBS1(i,j) = ns/SClassNum(Ys(i))*(-1);
            BBS1(j,i) = ns/SClassNum(Ys(i))*(-1);
        end
    end
end

YsColumn=singlelbs2multilabs(Ys,C);
BBS2 = YsColumn*YsColumn' - ones(ns,ns);

for i=1:ns
    BBS2(i,i) = ns - SClassNum(Ys(i));
end
BBS1=BBS1-ratio*BBS2;


if ~isempty(Yt0) && length(Yt0)==nt
    BBT1=zeros(nt,nt);
    for i=1:nt
            BBT1(i,i) = nt/TClassNum(Yt0(i))*(TClassNum(Yt0(i)));
        for j=i+1:nt
            if Yt0(i)==Yt0(j)
                            BBT1(i,j) = nt/TClassNum(Yt0(i))*(-1);
                            BBT1(j,i) = nt/TClassNum(Yt0(i))*(-1);
            end
        end
    end
    BBT1 = BBT1;

    Yt0Column=singlelbs2multilabs(Yt0,C);
    BBT2 = Yt0Column*Yt0Column' - ones(nt,nt);

    for i=1:nt
        BBT2(i,i) = nt - TClassNum(Yt0(i));
    end
    BBT1=BBT1-ratio*BBT2;
end

BB = blkdiag(BBS1,BBT1);
BB = BB/norm(BB,'fro');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

% DICD
if ~isempty(Yt0) && length(Yt0)==nt
    
    if strcmp(ker,'primal')
        [A,~] = eigs(X*M*X'+lambda*eye(m)+beta*X*BB*X',X*H*X',k,'SM');
        Z = A'*X;
     
    else
        K = kernel(ker,X,[],gamma);
        Ks=K(:,1:ns);
        [A,~] = eigs(K*M*K'+lambda*eye(n)+beta*K*BB*K',K*H*K',k,'SM');
        Z = A'*K;
    end
else
    if strcmp(ker,'primal')
        [A,~] = eigs(X*M*X'+lambda*eye(m)+beta*Xs*BB*Xs',X*H*X',k,'SM');
        Z = A'*X;
    else
        K = kernel(ker,X,[],gamma);
        Ks=K(:,1:ns);
        [A,~] = eigs(K*M*K'+lambda*eye(n)+beta*Ks*BB*Ks',K*H*K',k,'SM');
        Z = A'*K;
    end
end
fprintf('Algorithm DICD terminated!!!\n\n');
end

%% Convert single column labels to multi-column labels
function label=singlelbs2multilabs(y,nclass)
    L=length(y);
    label=zeros(L,nclass);
    for i=1:L
        label(i,y(i))=1;
    end
end



