% Transfer Feature Learning with Joint Distribution Adaptation.  
% M. Long, J. Wang, G. Ding, J. Sun, and P.S. Yu.
% IEEE International Conference on Computer Vision (ICCV), 2013.

% Contact: Mingsheng Long (longmingsheng@gmail.com)

clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 1.0;
options.ker = 'linear';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
options.eta = 1.0;          % DTLC eta
T = 10;

result = [];
source_domains = {'Amazon_fc7','Webcam_fc7'};
target_domains = {'Dslr_fc7', 'Amazon_fc7'};
for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    %% data preprocessing
    load(strcat('../data/Office-31_DeCAF7/',source));
	fts = double(fts);
	fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
	Xs = zscore(fts, 1);
	Xs = Xs';
    Ys = labels;
    load(strcat('../data/Office-31_DeCAF7/',target));
    fts = double(fts);
	fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
	Xt = zscore(fts, 1);
	Xt = Xt';
    Yt = labels;
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
result_aver = mean(result);
Result = [result;result_aver]*100
