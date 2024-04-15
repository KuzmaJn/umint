function uloha2()
range1 = ones(1, 10);
SPACE = [range1 * -500;
         range1 * 500];

for i = 1:10
oldPop = genrpop(50, SPACE);
bestFit = zeros(1, 500);
for j = 1:500
    fitVector = testfn3(oldPop);
    bestFit(j) = min(fitVector);
    bestTen = selbest(oldPop,fitVector,ones(1,2));
    
    %indices_to_keep = ~ismember(oldPop, bestTen, 'rows');
    %newPop = oldPop(indices_to_keep, :);
    P_v1 = seltourn(oldPop, fitVector, 24);
    P_v2 = selrand(oldPop, fitVector, 12);
    P_v3 = selsus(oldPop, fitVector, 12);
    P_v1 = crossov(P_v1, 4, 0);
    P_v2 = crossov(P_v2, 3, 0);
    P_v3 = crossov(P_v3, 2, 0);

    P_v1 = mutx(P_v1, 0.15, SPACE);
    P_v2 = muta(P_v2, 0.14, range1 * 0.5, SPACE);
    P_v3 = mutm(P_v3, 0.13, [range1*0.01; range1*2], SPACE);
    bestPop = [P_v1; P_v2; P_v3; bestTen];
    
   
    oldPop = bestPop;
end
selbest(bestPop, testfn3(oldPop), 1)
min(bestFit)
D_f = (1:500);
plot(D_f,bestFit);
title("GA pre Schwefelovu funkciu")
xlabel("Generacia")
ylabel("f(x)")
hold on
%legend("AutoUpdate","on")
end
end
