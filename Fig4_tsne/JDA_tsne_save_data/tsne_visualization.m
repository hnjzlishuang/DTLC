%%% tsne visulization
clc;
clear;
addpath('tSNE_matlab');

savefile1 = 'save_data\ZsZt_PIE07_PIE29_jda.mat';

load(savefile1);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xs=Zs';
Xt=Zt';
X=[Xs;Xt];

color1=[1,0,0];
color2=[0,0,1];
Y1=[repmat(color1,[length(Ys),1]);repmat(color2,[length(Yt),1])];

mappedX = tsne(X, Y1, 2,100,30);

savefile = ['save_tu\map_07_29_jda','.mat'];
save(savefile, 'mappedX','Ys','Yt');
