% načítanie a príprava dát pre - rozpoznavanie cislic 0-9, MNIST dataset
% data - XDataall - 784 x 4940 (vstupne data x vzorky)
clear

% nacitanie dat
load('datapiscisla_all.mat');

% Nulovanie premennych pre data
XTrain = [];
YTrain = [];
YTr = [];
XvalDat = [];
YvalDat = [];

trainLen = 1;        % index vzorky trenovacie data
valLen = 1;         % index vzorky testovacie data
imgSize = 28;       % vstupny rozmer obrazu
percento=60;        % percento trenovacich dat
len = length(YDataall); % dlzka dat

Train_idx = randperm(len,len/100*percento);   % nahodne vygenerovanie indexov dat pre trenovanie

for i = 1:10
    YTr = [YTr, i*ones(1,494)];
end

for i = 1:len
    xhelp = [];
    yhelp = [];
    y = 1;
    
    % rozbytie vektora do obrazu 28x28 po riadkoch
    for x = 1:784
        yhelp = [yhelp, XDataall(x, i)];
        if y == 28
            xhelp = [xhelp; yhelp];
            yhelp = [];
            y = 0;
        end
        y = y + 1;
    end
    
    % ak index vzorky je medzi trenovacimi datami
    if ismember(i, Train_idx)
        XTrain(:,:,1,trainLen) = xhelp;
        YTrain = [YTrain, YTr(i)];
        trainLen = trainLen + 1;
    else
        XvalDat(:,:,1,valLen) = xhelp;
        YvalDat = [YvalDat, YTr(i)];
        valLen = valLen + 1;
    end
end

% zmena na typ categorical
YTrain = categorical(YTrain');
YvalDat = categorical(YvalDat');

% figure
% idx = randperm(length(YvalDat),10);       % nahodne vygenerovanych 10 vzoriek
% 
% for i = 1:numel(idx)
%     subplot(2,5,i)    
%     imshow(XvalDat(:,:,:,idx(i)))
%     title(char(YvalDat(idx(i))))
%     drawnow
% end


layers = [imageInputLayer([28 28 1]);                                           % vstupná vrstva – obraz 28x28x1
          convolution2dLayer(7, 16, 'Stride',1);                             % 2D konvolúcia – 7 filtrov, rozmer 4x4, krok 1, doplnie 2 riadkov, stlpcov nulami
          reluLayer();                                                                % relu funkcia
          maxPooling2dLayer(2,'Stride',2)                                             % max pooling – 2x2, krok 2
          
          convolution2dLayer(2, 32, 'Padding', 0);                                      % 2D konvolúcia – 32 filtrov, rozmer 3x3
          batchNormalizationLayer
          reluLayer();                                                                % relu funkcia
          maxPooling2dLayer(2,'Stride',3);                                            % max pooling – 2x2, krok 3
          
          fullyConnectedLayer(10);                                                    % plne prepojená vrstva – 10 neurónov
          softmaxLayer();                                                             % softmax aktivačná funkcia
          classificationLayer()];                                                     % klasifikačná vrstva – 10 tried

options = trainingOptions('sgdm',...                                        % trénovací algoritmus sgdm alebo adam
                            'MaxEpochs',10, ...                             % počet trénovacích epoch
                            'InitialLearnRate',1e-2,...                     % počiat. krok učenia
                            'ValidationData', {XvalDat,YvalDat}, ...        % použitie validačných dát
                            'ValidationFrequency',50, ...                   % pri koľkej iterácii sa vykoná validácia
                            'MiniBatchSize',100, ...                        % veľkosť dávky obrazov pri trénovaní
                            'Shuffle', 'once', ...                          % premiešanie dát – raz pred trénovaním
                            'ExecutionEnvironment', 'gpu', ...              % trénovanie na cpu alebo gpu
                            'Plots','training-progress');                   % zobrazí sa proces trénovania v grafe




mAccuracy = zeros(1, 5);
for j = 1:1
    [net, info] = trainNetwork(XTrain, YTrain, layers, options);            % trenovanie siete
    classY = classify(net, XvalDat);
    mAccuracy(j) = info.ValidationAccuracy(:, end);
    close all;
end
fprintf('Best Test Accuracy: %.2f', max(mAccuracy))

figure(1)
idx = randperm(length(YvalDat),10);
for i = 1:10                                                                % vykreslenie klasifikacie 10 bodov
    subplot(5,2,i)
    imshow(XvalDat(:,:,:,idx(i)))
    title(['Predpoklad: ', char(classY(idx(i)))])
end

figure(2)
confusionchart(YvalDat,classY)                                              % kontingencna matica pre validacne data
title('Confusion matrix for validation data')