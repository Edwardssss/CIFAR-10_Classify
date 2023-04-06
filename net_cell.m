x = linspace(-10.0,10.0);
relu = max(x,0);
Prelu = max(0.25 * x,0);
plot(x,relu,'r');
hold on
plot(x,Prelu,'b');
hold on
legend('ReLU',"PReLU");
title('激活函数(Activation)');

axis([-2.0 5.0 -2.0 5.0])
set(gca, 'XGrid','on');  % X轴的网格
set(gca, 'YGrid','on');  % Y轴的网格