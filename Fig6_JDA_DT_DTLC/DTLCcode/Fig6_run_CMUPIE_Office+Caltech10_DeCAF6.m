clear all;

result = [];
%% CMU-PIE PIE05->PIE07
% Set algorithm parameters
options.k = 100;
options.alpha = 1.0;       % DTLC alpha
options.beta = 0.1;      % DTLC beta
options.eta= 1.0;          % DTLC eta
options.ker = 'primal';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % the number of (positive/negtive) data pair 
T = 10;

source_domains = {'PIE05'};
target_domains = {'PIE07'};
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

    %% JDA evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
       [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);

        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt); fprintf('JDA + NN = %0.4f\n',acc);
        Acc = [Acc;acc];
    end
	% the Classification Accuracy of JDA on office31(C->A) task
    result = [result;Acc(end)];
    fprintf('\n\n\n')
	
	%% DTLC w/o Label Consistency evaluation
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
        fprintf('DTLC + NN =%0.4f\n', acc);
        Acc = [Acc; acc(1)];
    end
	% the Classification Accuracy of DTLC w/o Label Consistency on office31(C->A) task
    result = [result; Acc(end)];
    fprintf('\n\n\n');
	
	%% DTLC evaluation
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
        Cls = predict_t; 
        
        acc = length(find(Cls==Yt)) / length(Yt); 
        fprintf('DTLC + NN =%0.4f\n', acc);
        Acc = [Acc; acc(1)];
    end
	% the Classification Accuracy of DELC on office31(C->A) task
    result   = [result; Acc(end)];
    fprintf('\n\n\n');
end

%% Office31 DeCAF6(C->A)
% Set algorithm parameters
options.beta = 1.0;      % DTLC beta
options.ker = 'linear';  % 'primal' | 'linear' | 'rbf'

source_domains = {'Caltech10'};
target_domains = {'Amazon'};
for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
   % Preprocess data 
    load(strcat('../../data/Office+Caltech10_DeCAF6/', source, '_decaf.mat'));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../../data/Office+Caltech10_DeCAF6/', target, '_decaf.mat'));
    Xt = fea';
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;   
	
    % 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n', acc);

    %% JDA evaluation
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
        Acc = [Acc;acc];
    end
	% the Classification Accuracy of JDA on office31(C->A) task
    result = [result;Acc(end)];
    fprintf('\n\n\n')
	
	%% DTLC w/o Label Consistency evaluation
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
	% the Classification Accuracy of DTLC w/o Label Consistency on office31(C->A) task
    result = [result; Acc(end)];
    fprintf('\n\n\n');
	
	%% DTLC evaluation
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
        Cls = predict_t; 
        
        acc = length(find(Cls==Yt)) / length(Yt); 
        fprintf('DTLC + NN =%0.4f\n', acc);
        Acc = [Acc; acc(1)];
    end
	% the Classification Accuracy of DELC on office31(C->A) task
    result = [result; Acc(end)];
    fprintf('\n\n\n');
end

y = reshape(result, 3, [])' * 100;
b = bar(y); %Draw a basic column chart
set(gca,'XTickLabel',{'C05->C07(CMU-PIE)', 'C->A(DeCAF6)'})
legend('JDA','DTLC w/o label consistency','DTLC');%Name each category
axis([0,3,40,100])%Set the range of values for x and y axes
ylabel('Classification Accuracy(%)');%Set the y-axis label
