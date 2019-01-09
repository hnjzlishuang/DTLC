function [M]=dist(X,Y,non) 
%% Construct a distance matrix of all variables
n = size(X,2);
L = repmat(sum(X'.*X',2)',n,1) + repmat(sum(X'.*X',2),1,n) - 2*(X'*X);

y=zeros(n,n);
%% if i,j are in the same class, y(i,j)=1
for i=1:n
    for j=i:n
        if Y(i)==Y(j)
            y(i,j) = 1;
            y(j,i) = 1;
        end
    end
end

%% for positive pair
Ls_same=L.*y;
 [~,pos] = sort(Ls_same,2,'descend');
i=reshape(repmat([1:n]',1,non)',1,non*n);
j=reshape(pos(:,[1:non])',1,non*n);
v=ones(1,non*n);
Ms_same=sparse(i,j,v,n,n);

%% for negtive pair
Ls_diff=L.*(ones(n,n)-y);
Ls_diff(find(Ls_diff==0))=NaN;
[~,pod] = sort(Ls_diff,2);
i=reshape(repmat([1:n]',1,non)',1,non*n);
j=reshape(pod(:,[1:non])',1,non*n);
v=-1*ones(1,non*n);
Ms_diff=sparse(i,j,v,n,n);
 
Ms = Ms_same+Ms_diff;

%% contruct matrix M
 M = sparse(n,n);
 [i,j] = find(Ms~=0);
 for m=1:size(i)
        i_m=i(m);
        j_m=j(m);
        M = M+Ms(i_m,j_m)*sparse([i_m,i_m,j_m,j_m],[i_m,j_m,i_m,j_m],[1,-1,-1,1],n,n);
 end
 

M = full(M);

 