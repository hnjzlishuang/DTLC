clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 1.0;
options.ker = 'linear';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10;

source_domains = {'Caltech10_decaf'};
target_domains = {'Amazon_decaf'};
result = [];

for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    % Preprocess data 
    load(strcat('../../data/Office+Caltech10_DeCAF6/', source, '.mat'));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../../data/Office+Caltech10_DeCAF6/', target, '.mat'));
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
    result = [result; Acc(end)];
    fprintf('\n\n\n');
end

% y =[43.1,45.0939,48.99657143;47.7 ,72.5490 ,77.2;89.8 90.3967 92.954;63.4,63.6848,64.8;56.8,60.6426,68.3533];
y = result * 100;
b = bar(y); %Draw a basic column chart
% set(gca,'XTickLabel',{'C->A(SURF)','C05->C07','C->A(DeCAF6)','I->V','A->D(OFFICE-31)'}) %Name each set of data
set(gca,'XTickLabel',{'C->A(DeCAF6)'})
legend('JDA','DTLC w/o label consistency','DTLC');%Name each category
axis([0,7,40,100])%Set the range of values for x and y axes
ylabel('classification accuracy(%)');%Set the y-axis label