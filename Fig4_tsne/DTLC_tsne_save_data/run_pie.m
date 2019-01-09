clear all;

% Set algorithm parameters
options.k = 100;
options.alpha = 1;       % DTLC alpha
options.beta = 0.1;      % DTLC beta
options.eta= 1;          % DTLC eta
options.ker = 'primal';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % the number of (positive/negtive) data pair 
T = 10;

srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};
    result = [];
for iData = 8 
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    %% data preprocessing
    load(strcat('../../data/CMU-PIE/',src));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../../data/CMU-PIE/',tgt));
    Xt = fea';
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;
    
    %% 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);

    %% DTLC evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        %% DTLC discriminative transfer feature learning
        [Z,A] = DTLC_DT(Xs,Xt,Ys,Cls,options);
        
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        
        Cls = knnclassify(Zt',Zs',Ys,1);
        
        %% DTLC label consistency
        options.NN=5;
        [label_t,predict_t,~]=DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls=predict_t; 
        
        acc = length(find(Cls==Yt))/length(Yt); 
        fprintf('DTLC+NN=%0.4f\n',acc);
        Acc = [Acc;acc(1)];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
    savefile = ['save_data\ZsZt_',char(srcStr{iData}),'_',char(tgtStr{iData}),'_dtlc.mat'];
    save(savefile, 'Zs', 'Zt','Ys','Yt');
end
Result=result*100