clear,clc,clf;
save_epoch = [5,10,15,20,25,30,35,40,45,50];
normal_accuracy = [60,63,64,63,64,63,62,62,62,61];
normal_plane = [76,76,64,64,73,67,69,73,78,75];
normal_car = [78,76,74,86,80,76,80,74,78,78];
normal_bird = [53,58,49,46,45,46,49,44,43,48];
normal_cat = [54,36,41,32,50,49,39,43,53,38];
normal_deer = [41,54,52,41,54,58,45,58,50,54];
normal_dog = [38,54,49,57,59,49,49,42,44,52];
normal_frog = [76,69,67,69,73,67,80,71,82,60];
normal_horse = [45,56,73,67,50,68,57,65,57,56];
normal_ship = [77,72,81,84,77,77,79,81,68,77];
normal_truck = [57,61,73,57,60,78,67,67,67,76];

figure(1);
subplot(3,1,1)
plot(save_epoch,normal_accuracy),xlabel("epoch"),ylabel("总体准确率/%");
axis([5,55,50,80]);
subplot(3,1,2)
hold on
plot(save_epoch,normal_plane);
plot(save_epoch,normal_car);
plot(save_epoch,normal_bird);
plot(save_epoch,normal_cat);
plot(save_epoch,normal_deer);
legend("plane","car","bird","cat","deer");
xlabel("epoch"),ylabel("准确率/%");
axis([5,55,20,100]);
subplot(3,1,3)
hold on
plot(save_epoch,normal_dog);
plot(save_epoch,normal_frog);
plot(save_epoch,normal_horse);
plot(save_epoch,normal_ship);
plot(save_epoch,normal_truck);
legend("dog","frog","horse","ship","truck");
xlabel("epoch"),ylabel("准确率/%");
axis([5,55,20,100]);

resnet_accuracy = [78,78,79,80,80,81,80,81,82,81];
resnet_plane = [89,87,85,89,85,85,85,91,89,87];
resnet_car = [92,90,90,92,92,90,90,90,92,88];
resnet_bird = [74,72,69,79,81,81,74,79,74,75];
resnet_cat = [50,64,61,63,57,49,61,65,65,61];
resnet_deer = [85,83,78,69,65,74,76,74,78,63];
resnet_dog = [79,69,66,67,76,71,74,76,74,76];
resnet_frog = [78,85,87,87,83,91,85,87,87,91];
resnet_horse = [85,82,79,79,79,84,82,89,82,84];
resnet_ship = [91,87,89,89,81,91,86,89,93,87];
resnet_truck = [82,83,83,87,83,89,87,85,88,83];

figure(2);
subplot(3,1,1);
plot(save_epoch,resnet_accuracy),xlabel("epoch"),ylabel("总体准确率/%");
axis([5,55,50,100]);
subplot(3,1,2)
hold on
plot(save_epoch,resnet_plane);
plot(save_epoch,resnet_car);
plot(save_epoch,resnet_bird);
plot(save_epoch,resnet_cat);
plot(save_epoch,resnet_deer);
legend("plane","car","bird","cat","deer");
xlabel("epoch"),ylabel("准确率/%");
axis([5,55,20,100]);
subplot(3,1,3)
hold on
plot(save_epoch,resnet_dog);
plot(save_epoch,resnet_frog);
plot(save_epoch,resnet_horse);
plot(save_epoch,resnet_ship);
plot(save_epoch,resnet_truck);
legend("dog","frog","horse","ship","truck");
xlabel("epoch"),ylabel("准确率/%");
axis([5,55,20,100]);

sgd_epoch =[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135];

sgd_accuracy = [79,81,84,84,84,84,84,84,84,85,84,84,84,82,84,86,86,86,86,86,86,86,82,84,84,87,87];
sgd_plane = [86,82,86,82,82,79,79,82,79,79,79,79,75,82,86,96,96,96,96,93,96,89,75,89,89,89,89];
sgd_car = [92,92,92,89,89,89,89,89,89,89,89,89,89,89,92,96,96,96,96,96,96,96,92,92,92,96,92];
sgd_bird = [69,72,78,75,75,72,72,72,72,72,72,72,75,75,75,78,75,78,81,81,81,75,72,75,78,81,81];
sgd_cat = [44,64,52,61,61,61,61,61,64,67,67,70,67,61,58,61,64,61,61,61,58,61,58,58,64,64,64];
sgd_deer = [77,85,81,85,81,81,77,77,81,85,77,85,62,85,85,85,85,85,85,85,85,85,85,85,85,88,88];
sgd_dog = [87,69,78,72,72,72,72,69,72,69,75,75,75,75,75,81,78,84,84,75,78,75,87,75,75,81,81];
sgd_frog = [77,77,83,83,83,83,83,83,80,80,80,86,80,83,80,83,83,83,83,80,83,83,80,80,83,91,88];
sgd_horse = [92,88,88,88,88,92,92,92,92,92,92,92,92,92,96,96,92,96,96,92,92,92,84,88,96,88,88];
sgd_ship = [90,90,90,90,90,90,87,87,87,87,90,90,90,90,96,96,96,96,96,96,96,93,93,96,93,100,100];
sgd_truck = [82,92,94,92,94,94,94,94,92,92,92,92,92,87,89,94,92,94,94,94,94,94,84,79,87,89,89];

figure(3);
subplot(3,1,1);
plot(sgd_epoch,sgd_accuracy),xlabel("epoch"),ylabel("总体准确率/%");
axis([5,135,70,100]);
subplot(3,1,2)
hold on
plot(sgd_epoch,sgd_plane);
plot(sgd_epoch,sgd_car);
plot(sgd_epoch,sgd_bird);
plot(sgd_epoch,sgd_cat);
plot(sgd_epoch,sgd_deer);
legend("plane","car","bird","cat","deer");
xlabel("epoch"),ylabel("准确率/%");
axis([5,135,20,100]);
subplot(3,1,3)
hold on
plot(sgd_epoch,sgd_dog);
plot(sgd_epoch,sgd_frog);
plot(sgd_epoch,sgd_horse);
plot(sgd_epoch,sgd_ship);
plot(sgd_epoch,sgd_truck);
legend("dog","frog","horse","ship","truck");
xlabel("epoch"),ylabel("准确率/%");
axis([5,135,20,100]);

