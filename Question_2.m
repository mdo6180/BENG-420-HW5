clear
close all

% labels: -1, -0.5, 0, 0.5, 1 represents non-Hispanic white, 
% non-Hispanic black, Mexican American, other, other Hispanic respectively
[y_train, x_train] = libsvmread('train.txt');
[y_test, x_test] = libsvmread('test.txt');

linear = svmtrain(y, x, sprintf('-s 0 -t 0'));
poly = svmtrain(y, x, sprintf('-s 0 -t 1'));
gaussian = svmtrain(y, x, sprintf('-s 0 -t 2'));
sigmoid = svmtrain(y, x, sprintf('-s 0 -t 3'));