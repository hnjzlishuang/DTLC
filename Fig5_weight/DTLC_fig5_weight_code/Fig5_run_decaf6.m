clear all;

% Set algorithm parameters
options.k = 100;
options.alpha = 1.0;       % DTLC alpha
options.beta = 1.0;      % DTLC beta
options.eta = 1.0;          % DTLC eta
options.ker = 'linear';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % the number of (positive/negtive) data pair 
T = 10;

source_domains = {'Caltech10','Caltech10','Amazon', 'Webcam', 'Dslr', 'Dslr'};
target_domains = {'Amazon','Webcam','Caltech10', 'Amazon','Caltech10', 'Webcam'};
result = [];
Acclamda=[];
for iData = 1:length(target_domains)
    source = char(source_domains{iData});
    target = char(target_domains{iData});
    options.data = strcat(source,'_vs_',target);
    
    %% data preprocessing
    load(['../../data/Office+Caltech10_DeCAF6/' source '_decaf.mat']);
    Xs = fea';
    meanXs = mean(Xs,2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
      Ys = gnd;
    load(['../../data/Office+Caltech10_DeCAF6/' target '_decaf.mat']);
    Xt = fea';
    meanXs = mean(Xt,2);
    Xt = bsxfun(@minus, Xt, meanXs);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;
    fprintf('DTLC:  data=%s alpha=%f beta=%f eta=%f\n', options.data, options.alpha, options.beta, options.eta);
         
    %% 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);
    
    %% DTLC evaluation
    Cls = [];
    Acc = [];
    AccforWeight=[];
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        %% DTLC discriminative transfer feature learning
        [Z,A] = DTLC_DT(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        
        % 1NN evaluation
        Cls = knnclassify(Zt',Zs',Ys,1);
        
        %% DTLC label consistency
        options.NN = 5;
        [label_t,predict_t,~,MeanAcc] = DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls=predict_t; 
        acc = length(find(Cls==Yt))/length(Yt); 
        fprintf('DTLC + NN =%0.4f\n', acc);
        Acc = [Acc;acc(1)];
        AccforWeight=[AccforWeight,MeanAcc];
    end
    Acclamda=[Acclamda; AccforWeight(:,end)];
    result = [result;Acc(end)];
    fprintf('\n\n\n'); 
end
result_aver=mean(result);
Result = [result;result_aver]*100;
lamda = reshape(Acclamda, 3, [])';
lamda = lamda * 100;
b = bar(lamda); %Draw a basic column chart
set(gca,'XTickLabel',{'C->A','C->W', 'A->C', 'W->A', 'D->C', 'D->W'}) %Name each set of data
lengd1=legend('1/3 samples with minimum ¦Ë','1/3 samples with middle ¦Ë','1/3 samples with maximum ¦Ë', 'Location','Best' );%Name each category
axis([0,7,70,105])%Set the range of values for x and y axes
ylabel('Accuracy(%)');%Set the y-axis label
xlabel('Task'); 