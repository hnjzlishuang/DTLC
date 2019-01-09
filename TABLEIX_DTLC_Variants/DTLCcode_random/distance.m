function [D]=distance(X,Y,non)
N = size(X,2);
y=zeros(N,N);
%% if i,j are in the same class, y(i,j)=1
for i=1:N
    for j=i:N
        if Y(i)==Y(j)
            y(i,j) = 1;
            y(j,i) = 1;
        end
    end
end

% C = length(unique(Ys));
% flag=0;
% for c = reshape(unique(Ys),1,C) 
%     e = zeros(ns,1);
%     e(Ys==c) = 1;
%     e(isinf(e)) = 0;
%     if flag==0
%         ys=e*e';
%         flag=1;
%     else
%         ys = ys + e*e';
%     end
% end

%% if i,j are in the different class, diff(i,j)=1 
diff=ones(N,N)-y;
%% Set diagonal 0
for i=1:N
    diff(i,i)=0;
end
for i=1:N
    y(i,i)=0;
end
%% for positive and negtive pairs
 D = sparse(N,N);
 
%% Randomly select non-positive sample points and negative sample points for each sample
 for t=1:N
     a = y(t,:);%Similar sample point
     b = diff(t,:);%different sample point
     [i,j] = find(a~=0);
     [m,n] = find(b~=0);
     %% for positive pair
     if size(i,2)>0
         count1 = randperm(size(i,2),non);
         for w=1:non
             i_1=i(count1(w));
             j_1=j(count1(w));
             D = D+sparse([i_1,i_1,j_1,j_1],[i_1,j_1,i_1,j_1],[1,-1,-1,1],N,N);
         end 
     end
     %% for negtive pair
     if size(m,2)>0
         count2 = randperm(size(m,2),non);
         for w=1:non
             i_2=m(count2(w));
             j_2=n(count2(w));
             D = D+sparse([i_2,i_2,j_2,j_2],[i_2,j_2,i_2,j_2],[-1,1,1,-1],N,N);
         end 
     end
 end

D = full(D);

 