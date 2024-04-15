clear, clc, close all
%suradnice x,y,z piatich skupin bodov
load databody

%vstupne a vystupne data na trenovanie neuronovej siete
datainnet= [data1; data2; data3; data4; data5;].';
dataoutnet= [ones(1, 50), zeros(1, 200); 
            zeros(1, 50), ones(1, 50), zeros(1,150); 
            zeros(1, 100), ones(1, 50), zeros(1, 100); 
            zeros(1, 150), ones(1, 50), zeros(1,50);
            zeros(1, 200), ones(1, 50);];
c = 1;

while c > 0.008
close all

%vytvorenie struktury siete
pocet_neuronov = 4 ;
net = patternnet(pocet_neuronov);

%parametre rozdelenia dat na trenovanie, validacne a testovanie
net.divideFcn='dividerand';
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.0;
net.divideParam.testRatio=0.2;

%nastavenie parametrov trenovania 
net.trainParam.goal = 10^(-10);       % ukoncovacia podmienka na chybu.
% net.trainParam.show = 15;           % frekvencia zobrazovania chyby
% net.trainParam.max_fail= 1;
net.trainParam.epochs = 436;        % maximalny pocet trenovacich epoch.
net.trainParam.min_grad = 1e-10;

%trenovanie NS
net = train(net,datainnet,dataoutnet);

%simulacia vystupu NS pre trenovacie data
%testovanie NS
outnetsim = sim(net,datainnet);

%percento neuspesne klasifikovanych bodov
c = confusion(dataoutnet,outnetsim)
end

%klasifikacia 5 novych bodov do tried
new_points = [0.5, 0.62, 0.88, 0.45, 0.82;
              0.2, 0.26, 0.94, 0.74, 0.63;
              0.70, 0.11, 0.59, 0.67, 0.54;];

y2 = net(new_points);

classes2 = vec2ind(y2)


% vykreslenie bodov podla skupin
h=figure;
plot3(data1(:,1),data1(:,2),data1(:,3),'b+')
hold on
plot3(data2(:,1),data2(:,2),data2(:,3),'co')
plot3(data3(:,1),data3(:,2),data3(:,3),'g*')
plot3(data4(:,1),data4(:,2),data4(:,3),'r*')
plot3(data5(:,1),data5(:,2),data5(:,3),'mx')

axis([0 1 0 1 0 1])
title('Data body')
xlabel('x')
ylabel('y')
zlabel('z')
hold on


for i = 1:5
    switch classes2(i)
        case 1
            plot3(new_points(1,i),new_points(2,i),new_points(3,i),'bo',...
                'MarkerSize', 10, 'MarkerFaceColor', '#FFFF00')
        case 2
            plot3(new_points(1,i),new_points(2,i),new_points(3,i),'co',...
                'MarkerSize', 10, 'MarkerFaceColor', '#FFFF00')
        case 3
            plot3(new_points(1,i),new_points(2,i),new_points(3,i),'go',...
                'MarkerSize', 10, 'MarkerFaceColor', '#FFFF00') 
        case 4
            plot3(new_points(1,i),new_points(2,i),new_points(3,i),'ro',...
                'MarkerSize', 10, 'MarkerFaceColor', '#FFFF00')  
        case 5
            plot3(new_points(1,i),new_points(2,i),new_points(3,i),'mo',...
                'MarkerSize', 10, 'MarkerFaceColor', '#FFFF00')  
    end
end
