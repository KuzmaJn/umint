% Pr�klad na aproxim�ciu nelin. funkcie pomocou NS typu
% MLP siet s 1 vstupom a 1 v�stupom
MSE_test = 1;
while MSE_test > 10^(-4)
close all
%pause
clear
clc
load datafun

% vytvorenie �trukt�ry NS 
% 1 vstup - x suradnica
% 1 skryt� vrstva s poctom neur�nov 17 s funkciou 'tansig'
% 1 v�stup s funkciou 'purelin' - y suradnica
% tr�novacia met�da - Levenberg-Marquardt
pocet_neuronov = 17;
net=fitnet(pocet_neuronov);

% indexove rozdelenie 

net.divideFcn='divideind';
net.divideParam.trainInd=indx_train;
net.divideParam.valInd=[];
net.divideParam.testInd=indx_test;


% Nastavenie parametrov tr�novania
net.trainParam.goal = 10^(-31);     % Ukoncovacia podmienka na chybu
net.trainParam.show = 5;        % Frekvencia zobrazovania priebehu chyby tr�novania
net.trainParam.epochs = 1500;      % maximalny pocet trenovacich epoch.

% % Tr�novanie NS
net=train(net,x,y);

% Simul�cia v�stupu NS
outnetsim = sim(net, x);
y1=y(indx_train);
y2=y(indx_test);

% vypocet chyby siete
% pre train data
train = outnetsim(indx_train) - y1;
MAE = mae(train)
train = train.^2;
SSE = sum(train)
MSE = mean(train)

% pre test data
test = outnetsim(indx_test) - y2;
MAE_test = mae(test)
test = test.^2;
SSE_test = sum(test)
MSE_test = mean(test)

end

% Vykreslenie do grafu 
figure
plot(x(indx_train),y1,'b+',x(indx_test),y2,'g*')
hold on
plot(x,outnetsim,'-r')
legend('Train data', 'Test data', 'MLP simulation')
