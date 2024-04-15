function uloha1()

function [y] = funkcia1D(x)
y = 0.2*x.^4+0.2*x.^3-4*x.^2+10;
end
a = linspace(-6, 6);
b = funkcia1D(a);

plot(a, b)
xlabel("x")
ylabel("y")
title("Horolezecky algoritmus")
hold on

step = 0.5; %0.2, 0.1
x = [-5 -1 1 5];
b2 = funkcia1D(x);
plot(x, b2,".y",'MarkerSize',20);

for i = 1:4
    x_l = x(i) - step;
    x_r = x(i) + step;
    y_x = funkcia1D(x(i));
    y_l = funkcia1D(x_l);
    y_r = funkcia1D(x_r);
    while y_x > y_l || y_x > y_r
        y_x = funkcia1D(x(i));
        y_l = funkcia1D(x_l);
        y_r = funkcia1D(x_r);
        if y_x > y_r
            x(i) = x_r;
        end
        if y_x > y_l
            x(i) = x_l;
        end
        x_l = x(i) - step;
        x_r = x(i) + step;
        plot(x(i),funkcia1D(x(i)), ".g",'MarkerSize',20)
    end
    
end
plot(min(x),funkcia1D(min(x)), "*r",'MarkerSize',20)
end

