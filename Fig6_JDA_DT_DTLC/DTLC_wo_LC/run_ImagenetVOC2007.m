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

source_domains = {'ImageNet', 'VOC2007'};
target_domains = {'VOC2007', 'ImageNet'};
result = [];

for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    %% data preprocessing
    load(strcat('../data/ImageNet+VOC2007/',source));
	data = double(data);
	Xs = data(:, 1:end-1);
	Xs = Xs ./ repmat(sum(Xs, 2), 1, size(Xs, 2));
	Xs = zscore(Xs, 1);
	Xs = Xs';
    Ys = data(:, end);
    load(strcat('../data/ImageNet+VOC2007/',target));
    data = double(data);
	Xt = data(:, 1:end-1);
	Xt = Xt ./ repmat(sum(Xt, 2), 1, size(Xt, 2));
	Xt = zscore(Xt, 1);
	Xt = Xt';
    Yt = data(:, end);
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
        fprintf('DTLC + NN = %0.4f\n', acc);
        Acc = [Acc; acc(1)];
    end
    result = [result; Acc(end)];
    fprintf('\n');
end
result_aver = mean(result);
fprintf('Average:\n');
Result = [result;result_aver]*100