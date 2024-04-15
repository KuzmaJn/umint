clear
load CTGdata.mat

% disp(' --------------- stlac klavesu --------------')
% pause

% vstupne a vystupne data na trenovanie neuronovej siete
 datainnet= NDATA';
 dataoutnet= zeros(3,size(typ_ochorenia,1)); % lebo sú 3 typy ochorenia
 for i=1:size(typ_ochorenia,1)
   dataoutnet(typ_ochorenia(i),i)=1;
 end
 pocet = 5;
for i=1:pocet
    % vytvorenie struktury siete
    pocet_neuronov = 25;
    net = patternnet(pocet_neuronov);
    % parametre rozdelenia dat na trenovanie, validacne a testovanie
    net.divideFcn='dividerand';
    net.divideParam.trainRatio=0.6;
    net.divideParam.valRatio=0.0;
    net.divideParam.testRatio=0.4;
    % nastavenie parametrov trenovania 
    net.trainParam.goal = 1e-15;      % ukoncovacia podmienka na chybu.
    net.trainParam.show = 20;           % frekvencia zobrazovania chyby
    net.trainParam.epochs = 300;        % maximalny pocet trenovacich epoch.
    net.trainParam.min_grad = 1e-10; % ukoncovacia podmienka pre min gradient
    %net.trainParam.max_fail=12;
    % trenovanie NS
    [net,traindata] = train(net,datainnet,dataoutnet);                      % vráti trénovanú sieť a dáta priebehu testovania
    vystup = net(datainnet);                                                % vypočítanie výstupu pre vsetky vstupne data
    outputstrain = net(datainnet(:,traindata.trainInd));                    % výstupy siete pre train data
    outputstest = net(datainnet(:,traindata.testInd));                      % výstupy siete pre test data
    trainTargets = dataoutnet(:,traindata.trainInd);                        % uloženie očakávaných výstupov zo vstupu pre train
    testTargets = dataoutnet(:,traindata.testInd);                          % uloženie očakávaných výstupov zo vstupu pre test
    performance = perform(net,dataoutnet,vystup);                           % porovnanie a teda vyčíslenie výkonu ns, dataoutnet očakavany, vystup vypočitany
    
    % percento neuspesne klasifikovanych dat
    [ctrain,cmTrain] = confusion(trainTargets,outputstrain);                % chybová miera z očakávanych a vypocitanych train vystupov
    [ctest,cmTest] = confusion(testTargets,outputstest);                    % chybová miera z očakávanych a vypocitanych test vystupov
    [c,cm] = confusion(dataoutnet,vystup);                                  % chybová miera z očakávanych a vypocitanych vystupov
                                                                            % senzitivita - meria schopnosť ns správne identifikovať pozitívne
                                                                            % príklady (reálne príklady pre podozrivé/patologické/normálne typy)
                                                                            % pomer počtu správne ident. k celkovému počtu pozitívnych príkladov
                                                                            % špecificita - meria schopnosť iden. negatívne príklady (nie sú ani pato, podozrivé, normálne)
                                                                            % pomer správne iden / celkový
    fprintf('%d. úspešne klasifikované dáta(train,test,all): %.4f  %.4f %.4f\n',i,100*(1-ctrain),100*(1-ctest), 100*(1-c));
    fprintf('Senzitivita a špecificita train = %.4f %.4f\n', cmTrain(2,2)/(cmTrain(2,2)+cmTrain(2,1)), cmTrain(1,1)/(cmTrain(1,1)+cmTrain(1,2)));
    fprintf('Senzitivita a špecificita test = %.4f %.4f\n', cmTest(2,2)/(cmTest(2,2)+cmTest(2,1)), cmTest(1,1)/(cmTest(1,1)+cmTest(1,2)));
    fprintf('Senzitivita a špecificita celkových dát = %.4f %.4f\n\n', cm(2,2)/(cm(2,2)+cm(2,1)), cm(1,1)/(cm(1,1)+cm(1,2)));
    % uloženie do matice výsledkov
    MaticaVysledkov(i,1) = 100*(1-ctrain); 
    MaticaVysledkov(i,2) = 100*(1-ctest);
    MaticaVysledkov(i,3) = 100*(1-c);
    i=i+1;
end
% klasifikacia vzoriek do tried 
vzorka_ochorenie(1,1)=typ_ochorenia(2,1);  %typ normalny
vzorka_ochorenie(2,1)=typ_ochorenia(1,1);  %podozrivy
vzorka_ochorenie(3,1)=typ_ochorenia(6,1);  %patologicky
vzorka(1,:)=NDATA(2,:);
vzorka(2,:)=NDATA(1,:);
vzorka(3,:)=NDATA(6,:);
vzorka2(:,1)=transpose(vzorka(1,:)); vzorka2(:,2)=transpose(vzorka(2,:)); vzorka2(:,3)=transpose(vzorka(3,:));

vystup_vzorka=net(vzorka2(:,1));
vystup_vzorka1=net(vzorka2(:,2));
vystup_vzorka2=net(vzorka2(:,3));

vystup_vzorka2(1,1) = vec2ind(vystup_vzorka);
vystup_vzorka2(2,1) = vec2ind(vystup_vzorka1);
vystup_vzorka2(3,1) = vec2ind(vystup_vzorka2);


Mmin = min(MaticaVysledkov);
Mmax = max(MaticaVysledkov);
priemer = mean(MaticaVysledkov);
fprintf('Usp test klasif(min,max,priem): %.4f %.4f %.4f \n',Mmin(2),Mmax(2), priemer(2));
fprintf('Usp tren klasif(min,max,priem): %.4f %.4f %.4f \n',Mmin(1),Mmax(1), priemer(1));
fprintf('Usp celk klasif(min,max,priem): %.4f %.4f %.4f \n',Mmin(3),Mmax(3), priemer(3));

