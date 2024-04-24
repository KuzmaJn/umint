% Loading dataset 
load datapiscisla_all.mat;

success_rate = zeros(6, 4)

for i = 1:6
% vytvorenie NS so skrytou vrstvou
hiddenLayerSize = 18; 
net = patternnet(hiddenLayerSize);

% nahodne delenie datasetu
net.divideFcn='dividerand';
net.divideParam.trainRatio=0.6;
net.divideParam.valRatio=0.1;
net.divideParam.testRatio=0.3;

% train parametre
net.trainParam.epochs = 250;
net.trainParam.show = 30; 
net.trainParam.max_fail = 20;    % max pocet validacnych chyb
net.trainParam.goal = 1e-7;

% trenovanie siete
[net, data_train] = train(net, XDataall, YDataall);                         % pouzitie GPU na trenovanie , 'UseGPU', 'yes'

% Evaluate performance on validation set
out_allsim = sim(net, XDataall);
out_trainsim = sim(net, XDataall(:,data_train.trainInd));
out_valsim = sim(net, XDataall(:,data_train.valInd));
out_testsim = sim(net, XDataall(:,data_train.testInd));
train_tgt = YDataall(:,data_train.trainInd);
test_tgt = YDataall(:,data_train.testInd);
val_tgt = YDataall(:,data_train.valInd);

c_train = confusion(train_tgt,out_trainsim);
c_test = confusion(test_tgt,out_testsim);
c_val = confusion(val_tgt,out_valsim);
c = confusion(YDataall, out_allsim);
success_rate(i, :) = [100*(1-c_train), 100*(1-c_val), 100*(1-c_test), 100*(1-c)];  

end

Mmin = min(success_rate);
Mmax = max(success_rate);
priemer = mean(success_rate);

fprintf('uspesnost train (min,max,priem): %.2f  %.2f  %.2f \n', Mmin(1), Mmax(1), priemer(1));
fprintf('uspesnost val (min,max,priem): %.2f  %.2f  %.2f \n', Mmin(2), Mmax(2), priemer(2));
fprintf('uspesnost test (min,max,priem): %.2f  %.2f  %.2f \n', Mmin(3), Mmax(3), priemer(3));
fprintf('uspesnost all (min,max,priem): %.2f  %.2f  %.2f \n',Mmin(4), Mmax(4), priemer(4));

input_vzorka_m = [XDataall(:,1), XDataall(:,908), XDataall(:,1035), XDataall(:,1553), XDataall(:,2004), XDataall(:,2703), XDataall(:,3233), XDataall(:,3677), XDataall(:,4200), XDataall(:,4940)];  
target_vystup = vec2ind([YDataall(:,1), YDataall(:,908), YDataall(:,1035), YDataall(:,1553), YDataall(:,2004), YDataall(:,2703), YDataall(:,3233), YDataall(:,3677), YDataall(:,4200), YDataall(:,4940)])

ns_vystup_m = net(input_vzorka_m);
ns_vystup = vec2ind(ns_vystup_m)