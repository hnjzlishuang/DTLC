clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 1.0;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
options.eta = 1.0;          % DTLC eta
T = 10;

source_domains = {'Caltech10_decaf',  'Amazon_decaf'};
target_domains = { 'Dslr_decaf', 'Caltech10_decaf'};
result = [];

for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    % Preprocess data 
    load(strcat('../data/Office+Caltech10_DeCAF6/', source, '.mat'));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../data/Office+Caltech10_DeCAF6/', target, '.mat'));
    Xt = fea';
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;   
	
    % 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n', acc);

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