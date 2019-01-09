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

source_domains = {'Caltech10'};
target_domains = {'Webcam'};
result = [];
for i=1:7
    if i==1 options.alpha=0.01;
    elseif i==2 options.alpha=0.05;
    elseif i==3 options.alpha=0.1;
    elseif i==4 options.alpha=0.5;
    elseif i==5 options.alpha=1;
    elseif i==6 options.alpha=5;
    elseif i==7 options.alpha=10;
    end
    for iData = 1:length(target_domains)
        source = char(source_domains{iData});
        target = char(target_domains{iData});
        options.data = strcat(source,'_vs_',target);

        %% data preprocessing
        load(strcat('../data/Office+Caltech10_DeCAF6/', source, '_decaf.mat'));
        Xs = fea';
        meanXs = mean(Xs, 2);
        Xs = bsxfun(@minus, Xs, meanXs);
        Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
        Ys = gnd;
        load(strcat('../data/Office+Caltech10_DeCAF6/', target, '_decaf.mat'));
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

            %% DTLC label consistency
            options.NN = 5;
            [label_t,predict_t,~] = DTLC_LC(Zs',Ys,Zt',Yt,options,Cls);
            Cls = predict_t; 

            acc = length(find(Cls==Yt)) / length(Yt); 
            fprintf('DTLC + NN =%0.4f\n', acc);
            Acc = [Acc; acc(1)];
        end
        result = [result; Acc(end)];
        fprintf('\n');
    end
end