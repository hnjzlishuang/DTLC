clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 0.1;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
options.beta = 1;     %control manifold distance
options.ratio = 1;
T = 10;

result = [];
srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};
for iData =8
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    % Preprocess data using L2-norm
    load(strcat('../data/CMU-PIE/',src));
    Xs = fea';
    Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../data/CMU-PIE/',tgt));
    Xt = fea';
    Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    Yt = gnd;
    
    % 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);

    % DICD evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = DICD(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);

        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt); fprintf('DICD+NN=%0.4f\n',acc);
        Acc = [Acc;acc];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
    savefile = ['save_data\ZsZt_',char(srcStr{iData}),'_',char(tgtStr{iData}),'_dicd.mat'];
    save(savefile, 'Zs', 'Zt','Ys','Yt');
end
result
