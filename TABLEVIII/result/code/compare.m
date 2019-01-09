%correct
srcStr = {'amazon_fc7_dslr_fc7','amazon_fc7_webcam_fc7','dslr_fc7_amazon_fc7','dslr_fc7_webcam_fc7','webcam_fc7_amazon_fc7','webcam_fc7_dslr_fc7'};  
tgtStr = {'dslr_fc7','webcam_fc7','amazon_fc7','webcam_fc7','amazon_fc7','dslr_fc7'};
%correct1 DTLC wrong,DTLC w/o LC right
%correct2 DTLC right,DTLC w/o LC wrong
correct1 = [];
correct2 = [];
result = [];
for iData = 1:6
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    load(['../save_2/Y_' src '.mat']);
    load(['../save_3/Y_' src '.mat']);
    load(['../target/' tgt '.mat']);
    Yt = labels;
    y_DT=Cls2;
    y_DTLC=Cls3;
    w_DT = find(y_DT ~= Yt);
    w_DTLC = find(y_DTLC ~= Yt);
    
    same = intersect(w_DTLC,w_DT);

    %DTLC wrong,DTLC w/o LC right
    num1 = length(w_DTLC) -length(same);
    %DTLC right,DTLC w/o LC wrong
    num2 = length(w_DT) -length(same); 
    correct1 = [correct1;num1];
    correct2 = [correct2;num2];
end
correct1
fprintf('========================================================');
correct2    
fprintf('========================================================');
