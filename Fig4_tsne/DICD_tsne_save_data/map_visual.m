%%% map
clc;
clear;

savefile1 = 'save_tu\map_07_29_dicd.mat';
load(savefile1);
count_Ys = length(Ys);
count_Yt = length(Yt);
count_Y = count_Ys + count_Yt;

%red
color1=[1,0,0];
color2=[0,0,1];
Y1=[repmat(color1,[length(Ys),1]);repmat(color2,[length(Yt),1])];

%Display scatter plot (maximally first three dimensions)

scatter(mappedX(1:end,1), mappedX(1:end,2), 9, Y1, 'filled');
drawnow



    
