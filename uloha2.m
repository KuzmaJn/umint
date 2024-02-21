function uloha2()
range1 = ones(1, 10);
SPACE = [range1 * -500;
         range1 * 500];
newPop = genrpop(50, SPACE);
fitVector = testfn3(newPop);
bestPup = selbest(newPop,fitVector,1);
end