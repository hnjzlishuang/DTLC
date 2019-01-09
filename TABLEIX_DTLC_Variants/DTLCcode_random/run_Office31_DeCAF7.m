clear all;

% Set algorithm parameters
options.k = 100;
options.alpha = 1.0;       % DTLC alpha
options.beta = 1.0;      % DTLC beta
options.eta = 1.0;          % DTLC eta
options.ker = 'primal';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % the number of (positive/negtive) data pair 
T = 10;

source_domains = {'Amazon_fc7', 'Webcam_fc7'};
target_domains = {'Dslr_fc7', 'Amazon_fc7'};
result = [];

for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    %% data preprocessing
    load(strcat('../../data/Office-31_DeCAF7/',source));
	fts = double(fts);
	fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
	Xs = zscore(fts, 1);
	Xs = Xs';
    Ys = labels;
    load(strcat('../../data/Office-31_DeCAF7/',target));
    fts = double(fts);
	fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
	Xt = zscore(fts, 1);
	Xt = Xt';
    Yt = labels;
    fprintf('DTLC-random:  data=%s alpha=%f beta=%f eta=%f\n', options.data, options.alpha, options.beta, options.eta);
	
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
        
        %% DTLC label consistency
        options.NN = 5;
        [label_t,predict_t,~] = DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls=predict_t; 
        
        acc = length(find(Cls==Yt)) / length(Yt); 
        fprintf('DTLC-random = %0.4f\n', acc);
        Acc = [Acc; acc(1)];
    end
    result = [result; Acc(end)];
    fprintf('\n');
end
result_aver = mean(result);
fprintf('Average:\n');
Result = [result;result_aver]*100