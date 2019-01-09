% Transfer Feature Learning with Joint Distribution Adaptation.  
% M. Long, J. Wang, G. Ding, J. Sun, and P.S. Yu.
% IEEE International Conference on Computer Vision (ICCV), 2013.

% Contact: Mingsheng Long (longmingsheng@gmail.com)

clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 0.1;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
options.eta = 1.0;          % DTLC eta
T = 10;

result = [];
srcStr = {'PIE05','PIE05'};
tgtStr = {'PIE27','PIE29'};
for iData = 1:2
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(strcat('../../data/CMU-PIE/',src));
    Xs = fea';
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../../data/CMU-PIE/',tgt));
    Xt = fea';
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Yt = gnd;
    
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
        
        % label consistency
        options.NN = 5;
        [label_t,predict_t,~] = DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls = predict_t;
        
        acc = length(find(Cls==Yt))/length(Yt); fprintf('JDA-LC+NN=%0.4f\n',acc);
        Acc = [Acc;acc];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
end
result_aver=mean(result);
Result=[result;result_aver]*100
