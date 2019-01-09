clc
clear;
y=[43.1,45.0939,48.99657143;47.7 ,72.5490 ,77.2;89.8 90.3967 92.954;63.4,63.6848,64.8;56.8,60.6426,68.3533];
b=bar(y); %Draw a basic column chart
set(gca,'XTickLabel',{'C->A(SURF)','C05->C07','C->A(DeCAF6)','I->V','A->D(OFFICE-31)'}) %Name each set of data
legend('JDA','DTLC w/o label consistency','DTLC');%Name each category
axis([0,7,40,100])%Set the range of values for x and y axes
ylabel('classification accuracy(%)');%Set the y-axis label