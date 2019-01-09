function [label_t,predict_t,Accuracy,MeanAcc]=DTLC_LC(Xs,Ys,Xt,Yt,options,yt0)
Nt=length(Yt);
Ns=length(Ys);
class_set=unique(Ys);
nClass=length(class_set);

%% Compute pseudo label matrix
sparse_label=sparse(1:size(yt0,1),yt0,ones(1,size(yt0,1)),size(yt0,1),length(unique(Yt)));
label_t0=full(sparse_label);

acc_t0=mean(yt0==Yt)*100;

%% Dimensionality Reduction PCA
dim=100;
[~,score,latent,~]=pca([Xs;Xt]);
Xs=score(1:length(Ys),1:dim);
Xt=score(length(Ys)+1:end,1:dim);

%% Compute Xt High-dimensional Kernel matrix HDK
%%% options.Kernel: 'linear' | 'poly' | 'rbf' 
options.Kernel='linear';
if strcmp(options.Kernel,'linear')
        options.KernelParam=0;
    elseif strcmp(options.Kernel,'poly')
        options.KernelParam=5;
    elseif strcmp(options.Kernel,'rbf')
        options.KernelParam=50;
end
HDK=calckernel(options,Xt);

%% Compute graph Laplacian matrix
options.GraphWeights='binary';
options.GraphDistanceFunction='euclidean';
options.LaplacianNormalize=0;
options.LaplacianDegree=1;
L=laplacian(options,Xt);%it picks the neighbors from 2nd to NN+1th

%% reweighting scheme
options.Kfold=2;

%% Train domain separator
disp('Train domain separator ...');

%% Compute weight matrix from target domain to class-wise domain classifier
T=zeros(Nt,1);
for class=1:nClass
    Ntc=sum(yt0==class_set(class));
    if Ntc == 0
        ratio_Nt_Ntc = 0;
    else
        ratio_Nt_Ntc = Nt/Ntc;
    end
    [T_tmp,~,~,~]=dormain_separator_train(Xs(Ys==class_set(class),:),Xt,options);
    T=T+T_tmp.*(yt0==class_set(class))*ratio_Nt_Ntc;
end



%% Compute weight matrix from source domain to class-wise domain classifier
S=zeros(Ns,1);
for class=1:nClass
    [S_tmp,~,~,~]=dormain_separator_train(Xt,Xs(Ys==class_set(class),:),options);
    Nsc=sum(Ys==class_set(class));
    S(find(Ys==class_set(class)))=S_tmp*(Ns/Nsc);    
end

%% Divide three folds according to the weight from big to small and compute average accuracy seperately
[~,rank]=sort(T);
part=fix(Nt/3);
rank1=rank(1:part,:);
rank2=rank((part+1):(2*part),:);
rank3=rank((2*part+1):(3*part),:);
AccTab=(Yt==yt0);
Acc1=mean(AccTab(rank1));
Acc2=mean(AccTab(rank2));
Acc3=mean(AccTab(rank3));
MeanAcc=[Acc1;Acc2;Acc3];
 
%% Train target classifier
acc_v_final=0;
predict_t_final=[];
for i=1:11
    options.rho=10^(i-6);    %rho -5~5 
    Weight=(HDK*(diag(T)+options.rho*L)*HDK+0.01*eye(size(Xt,1)))\(HDK*bsxfun(@times,T,label_t0));
    label_t=HDK*Weight;
    HDSK=Xs*Xt';
    label_s=HDSK*Weight;
    if nClass==2
        predict_t=sign(label_t);
        predict_s=sign(label_s);
    else
        [~,predict_t] = max(label_t,[],2);
        [~,predict_s] = max(label_s,[],2);
    end
    acc(i,1)=mean(predict_t==Yt);
    acc_v(i,1)=S'*(predict_s==Ys);
    if acc_v_final< S'*(predict_s==Ys)
       predict_t_final=predict_t;
       acc_v_final=S'*(predict_s==Ys);
    end;
end
predict_t=predict_t_final;
acc_best=100*max(max(acc));
acc_da=100*acc(acc_v==max(max(acc_v)));
disp(['Accuracy with domain adaptation: ',num2str(acc_da(1,1)),'%/',...
    num2str(acc_best),'%(Best)/',num2str(acc_t0),'%(No adaption)']);
Accuracy=[acc_t0(1);acc_da(1,1);acc_best];
