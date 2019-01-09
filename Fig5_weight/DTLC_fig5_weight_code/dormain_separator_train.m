function [S,out_t,idx_t,acc_ds]=dormain_separator_train(Xs,Xt,options)
eta=options.eta;
Y=[ones(size(Xt,1),1);-ones(size(Xs,1),1)];
X=[Xt;Xs];

% Model selection by K-fold CV
Kfold=options.Kfold;
C_set=10.^[-4:0];
idx=kfoldcv(Y,Kfold);
acc=[];
for fold=1:Kfold
    xte=X(idx==fold,:);
    yte=Y(idx==fold);
    xtr=X(idx~=fold,:);
    ytr=Y(idx~=fold);
    
    for iC=1:length(C_set)
        str=['-t 0 -q -c ',num2str(C_set(iC))];
        svmModel = svmtrain(ytr, xtr, str);
        out= svmpredict(yte, xte, svmModel);
        acc(iC,fold)=mean(out==yte);
    end
end
acc_mean=mean(acc,2);
[row,~]=find(acc_mean==max(max(acc_mean)));
C_best = C_set(row(1,1));

str=['-t 0 -q -c ',num2str(C_best)];
model_ds=svmtrain(Y,X, str);
[~, acc_ds, out_all]=svmpredict(Y,X,model_ds);
out_t=out_all(1:size(Xt,1));
[~, idx_t]=sort(out_all(1:size(Xt,1)));

S=1./(1+exp(eta*(out_t-1)));

function idx=kfoldcv(y,K)

idx=zeros(size(y));
label_set=unique(y);
m=length(label_set);

for i=1:m
    idx_sub=(y==label_set(i));
    idx_cv_sub=crossvalind('Kfold', sum(idx_sub), K);
    idx(idx_sub)=idx_cv_sub;
end
