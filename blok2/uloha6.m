clear
load CTGdata.mat

% in and out data na trenovanie NS
datainnet= NDATA';
dataoutnet= zeros(3,size(typ_ochorenia,1));                                 % matica 3x2126 s hodnotou 1 tam v riadku s cislom podla typu ochorenia
                                                                            % matica s ocakavanymi vystupmi output = tgt
for i=1:size(typ_ochorenia,1)
    dataoutnet(typ_ochorenia(i),i)=1;
end

pocet = 5;                                                                  %spustenie 5x podla zadania

for i = 1:pocet
    % vytvorenie struktury siete
    pocet_neuronov = 23;
    net = patternnet(pocet_neuronov);
    
    % nahodne rozdelenie dat na train, val a test
    net.divideFcn='dividerand';
    net.divideParam.trainRatio=0.6;
    net.divideParam.valRatio=0.0;
    net.divideParam.testRatio=0.4;
    
    % nastavenie parametrov trenovania 
    net.trainParam.goal = 1e-15;                                            % ukoncovacia podmienka na chybu.
    net.trainParam.show = 20;                                               % frekvencia zobrazovania chyby
    net.trainParam.epochs = 300;                                            % maximalny pocet trenovacich epoch.
    net.trainParam.min_grad = 1e-10;                                        % ukoncovacia podmienka pre min gradient

    % trenovanie NS
    [net,data_train] = train(net,datainnet,dataoutnet);                     % vráti trénovanú sieť a dáta priebehu testovania
    output_all = net(datainnet);                                                % vypočítanie výstupu pre vsetky vstupne data
    output_train = net(datainnet(:,data_train.trainInd));                   % výstupy siete pre train data
    output_test = net(datainnet(:,data_train.testInd));                     % výstupy siete pre test data
    train_tgt = dataoutnet(:,data_train.trainInd);                          % uloženie očakávaných výstupov zo vstupu pre train
    test_tgt = dataoutnet(:,data_train.testInd);                            % uloženie očakávaných výstupov zo vstupu pre test
    performance = perform(net,dataoutnet,output_all);                           % porovnanie a teda vyčíslenie výkonu ns, dataoutnet očakavany, vystup vypočitany
    
    % percento neuspesne klasifikovanych dat
    [c_train,cmatrix_train] = confusion(train_tgt,output_train);                        % chybová miera z očakávanych a vypocitanych train vystupov
    [c_test,cmatrix_test] = confusion(test_tgt,output_test);                            % chybová miera z očakávanych a vypocitanych test vystupov
    [c,cm] = confusion(dataoutnet,output_all);                                              % chybová miera z očakávanych a vypocitanych vystupov
                                                                                        
                                                                                        % senzitivita - meria schopnosť ns správne identifikovať pozitívne príklady 
    sensMartix = [cmatrix_train(2,2) / (cmatrix_train(2,2) + cmatrix_train(2,1));       % (reálne príklady pre podozrivé/patologické/normálne typy)  vzorec: TP / (TP + FN)
                  cmatrix_test(2,2) / (cmatrix_test(2,2) + cmatrix_test(2,1));          % špecificita - meria schopnosť iden. negatívne príklady (nie sú ani pato, podozrivé, normálne) vzorec: TN / (TN + FP)
                  cm(2,2) / (cm(2,2) + cm(2,1)) ];                                      
    specMatrix = [cmatrix_train(1,1) / (cmatrix_train(1,1) + cmatrix_train(1,2));
                  cmatrix_test(1,1) / (cmatrix_test(1,1) + cmatrix_test(1,2));
                  cm(1,1) / (cm(1,1) + cm(1,2)) ];
    success_rate(i, :) = [100*(1-c_train), 100*(1-c_test), 100*(1-c)];                  %matica kde sa zapisuje uspesnost NS ktora je potom pouzita na vypocet min max a priemeru


    fprintf('%d. trenoavanie \nuspesnost (train, test, all): %.4f  %.4f  %.4f \n',i, success_rate(i,:));
    fprintf('Senzitivita (train, test, all) = %.4f  %.4f  %.4f \n', sensMartix);
    fprintf('Specificita (train, test, all) = %.4f  %.4f  %.4f \n\n', specMatrix);

    % uloženie do matice výsledkov

end

% vyber vzoriek
vzorka_ochorenie = [typ_ochorenia(2,1), typ_ochorenia(195,1), typ_ochorenia(6,1)];  %typ normalny, podozrivy, patologicky
vzorka = [NDATA(2,:)', NDATA(195,:)', NDATA(6,:)';];

fprintf('Spravna klasifikacia: %d  %d  %d \n', vzorka_ochorenie)

%klasifikacia vzoriek
vystup_vzorka=net(vzorka);
vystup_vzorky = vec2ind(vystup_vzorka);
fprintf('NS klasifikacia:  %d  %d  %d \n\n', vystup_vzorky)

Mmin = min(success_rate);
Mmax = max(success_rate);
priemer = mean(success_rate);

fprintf('uspesnost train (min,max,priem): %.4f  %.4f  %.4f \n', Mmin(1), Mmax(1), priemer(1));
fprintf('uspesnost test (min,max,priem): %.4f  %.4f  %.4f \n', Mmin(2), Mmax(2), priemer(2));
fprintf('uspesnost all (min,max,priem): %.4f  %.4f  %.4f \n',Mmin(3), Mmax(3), priemer(3));