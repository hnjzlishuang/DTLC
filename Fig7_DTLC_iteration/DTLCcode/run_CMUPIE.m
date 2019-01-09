clear all;

% Set algorithm parameters
options.k = 100;
options.alpha = 1.0;       % DTLC alpha
options.beta = 0.1;      % DTLC beta
options.eta= 1.0;          % DTLC eta
options.ker = 'primal';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % the number of (positive/negtive) data pair 
T = 10;

source_domains = {'PIE05','PIE07','PIE09','PIE27'};
target_domains = {'PIE07','PIE05','PIE05','PIE05'};
result = [];
for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    %% data preprocessing
    load(strcat('../../data/CMU-PIE/',source));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../../data/CMU-PIE/',target));
    Xt = fea';
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;
    fprintf('DTLC:  data=%s alpha=%f beta=%f eta=%f\n', options.data, options.alpha, options.beta, options.eta);
	
    %% 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n', acc);

    %% DTLC evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        %% DTLC discriminative transfer feature learning
        [Z,A] = DTLC_DT(Xs,Xt,Ys,Cls,options);
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        
		% 1NN evaluation
        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt)) / length(Yt); 
        Acc = [Acc, acc];
        
        %% DTLC label consistency
        options.NN = 5;
        [label_t,predict_t,~] = DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls=predict_t; 
        
        acc = length(find(Cls==Yt)) / length(Yt); 
        fprintf('DTLC + NN=%0.4f\n', acc);
        Acc = [Acc, acc];
    end
    result = [result; Acc];
    fprintf('\n');
end
result