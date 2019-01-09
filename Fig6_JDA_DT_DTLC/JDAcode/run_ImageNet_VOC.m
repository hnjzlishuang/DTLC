clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 1.0;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10;

srcStr = {'ImageNet','VOC2007'};
tgtStr = {'VOC2007','ImageNet',};
result = [];
for iData = 1:2
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);

    % Preprocess data using Z-score
    load(['../data/' src '.mat']);
    data = double(data);
    Xs = data(:,1:end-1);
    
    Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));
    Xs = zscore(Xs,1);
    
    Xs = Xs';
    Ys = data(:,end);
    load(['../data/' tgt '.mat']);
    data = double(data);
    Xt = data(:,1:end-1);
    
    Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2));
    Xt = zscore(Xt,1);
    
    Xt = Xt';
    Yt = data(:,end);

    % 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);
    
    % JDA evaluation
    Cls = [];
    Acc = [];
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);

        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt); fprintf('JDA+NN=%0.4f\n',acc);
        Acc = [Acc;acc(1)]; 
    end
    
    result = [result;Acc(end)];
    fprintf('\n\n\n');
end
result_aver=mean(result);
Result=[result;result_aver]*100
