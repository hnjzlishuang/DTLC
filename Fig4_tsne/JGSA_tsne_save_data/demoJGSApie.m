% Joint Geometrical and Statistical Alignment for Visual Domain Adaptation.
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
% Jing Zhang, Wanqing Li, Philip Ogunbona.

clear;close all;
datapath = 'data/';
% Set algorithm parameters
options.k = 30;             % subspace base dimension
options.ker = 'primal';     % kernel type, default='linear' options: linear, primal, gauss, poly

options.T = 10;             % #iterations, default=10

options.alpha= 1;           % the parameter for subspace divergence ||A-B||
options.mu = 1;             % the parameter for target variance
options.beta = 0.1;         % the parameter for P and Q (source discriminaiton)
options.gamma = 2;          % the parameter for kernel
srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};
  
results = [];
for iData = 8
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        options.data = strcat(src,'-vs-',tgt);
        fprintf('Data=%s \n',options.data);


        load and preprocess data  
        load([datapath '../../data/CMU-PIE/' src '.mat']);
        fea = double(fea);
        Xs = fea ./ repmat(sum(fea,2),1,size(fea,2)); 
        Ys = gnd;
        Xs = zscore(Xs);
         Xs = normr(Xs)';
        load([datapath '../../data/CMU-PIE/' tgt '.mat']);
        fea = double(fea);
        Xt = fea ./ repmat(sum(fea,2),1,size(fea,2)); 
        Yt = gnd;
        Xt = zscore(Xt);
         Xt = normr(Xt)';

    Cls = knnclassify(Xt',Xs',Ys,1); 
    acc = length(find(Cls==Yt))/length(Yt); 
    fprintf('acc=%0.4f\n',full(acc));

    Yt0 = Cls;
    [Zs, Zt, A, Att] = JGSA(Xs, Xt, Ys, Yt0, Yt, options);
    Cls = knnclassify(Zt',Zs',Ys,1); 
    acc = length(find(Cls==Yt))/length(Yt); 
    results = [results;acc];
    savefile = ['save_data\ZsZt_',char(srcStr{iData}),'_',char(tgtStr{iData}),'_jgsa.mat'];
    save(savefile, 'Zs', 'Zt','Ys','Yt');
end
results*100
