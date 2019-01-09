%%% tsne visulization
clc;
clear;
datapath = 'data/';
addpath('tSNE_matlab');
    % Preprocess data using L2-norm
    load([datapath '31/PIE07.mat']);
    Xs = fea';
    % Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
    % mean zero
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    % L2 normalization
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load([datapath '31/PIE29.mat']);
    Xt = fea';
    % Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;
% savefile1 = 'save_data\ZsZt_PIE05_PIE07_JDA.mat';
% savefile2 = 'save_data\ZsZt_PIE05_PIE07_DICDS.mat';
% savefile3 = 'save_data\ZsZt_PIE05_PIE07_DICD.mat';
% srcStr = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
% tgtStr = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10',    'amazon','dslr','Caltech10','amazon','webcam'};
% savefile1 = 'save_data\ZsZt_PIE07_PIE29.mat';
% savefile2 = 'save_data\ZsZt_Caltech10_webcam_DICDS.mat';
% savefile3 = 'save_data\ZsZt_Caltech10_webcam_DICD.mat';
% savefile4 = 'save_data\ZsZt_Caltech10_webcam_TCA.mat';

% savefile1 = 'save_data\ZsZt_MNIST_vs_USPS_JDA.mat';
% savefile2 = 'save_data\ZsZt_MNIST_vs_USPS_DICD.mat';

% load(savefile1);



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%old
Xs=Xs';
Xt=Xt';
X=[Xs;Xt];
% Y=[Ys;Yt];
%red
color1=[1,0,0];
color2=[0,0,1];
Y1=[repmat(color1,[length(Ys),1]);repmat(color2,[length(Yt),1])];

% load(savefile2);
% Xs_JDA=Zs';
% Xt_JDA=Zt';
% X_JDA=[Xs_JDA;Xt_JDA];
% mappedX = tsne(X_JDA, Y1, 2,100,30);

% mappedX = tsne(X, labels, 2, init_dims, perplexity)
% mappedXs = tsne(Xs_LMDA, Ys, 2,30,30);
% mappedXt = tsne(Xt_LMDA, Yt, 2,30,30);
mappedX = tsne(X, Y1, 2,100,30);
%30

    savefile = ['save_tu\map_07_29','_jgsa.mat'];
    save(savefile, 'mappedX','Ys','Yt');
% Display scatter plot (maximally first three dimensions)
%     scatter(ydata(1:157,1), ydata(1:157,2), 9, labels, 'filled');
%     axis tight
%     axis off
%     drawnow
% gscatter(mappedX(:,1),mappedX(:,2),Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Xs=Zs';
% Xt=Zt';
% ys=[];
% xs=[];
% yt=[];
% xt=[];
% Class=10;
% for i=1:Class
%     y=Ys(find(Ys==i),:);
%     x=Xs(find(Ys==i),:);
%     ys=[ys;y];
%     xs=[xs;x];
%     
%     y=Yt(find(Yt==i),:);
%     x=Xt(find(Yt==i),:);
%     yt=[yt;y];
%     xt=[xt;x];
%     
% end
% x=[xs;xt];
% color1=[1,0,0];
% color2=[0,0,1];
% y1=[repmat(color1,[length(ys),1]);repmat(color2,[length(yt),1])];
% mappedX = tsne(x, y1, 2,100,30);
