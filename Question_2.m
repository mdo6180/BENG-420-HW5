clear
close all

% Question 2:--------------------------------------------------------------

% a:----------------------------------------------------------------------

% labels: -1, -0.5, 0, 0.5, 1 represents non-Hispanic white, 
% non-Hispanic black, Mexican American, other, other Hispanic respectively
[y_train, x_train] = libsvmread('train.txt');
[y_test, x_test] = libsvmread('test.txt');

% -------------------------------------------------------------------------

% b:-----------------------------------------------------------------------
linear = svmtrain(y_train, x_train, sprintf('-s 0 -t 0'));
poly = svmtrain(y_train, x_train, sprintf('-s 0 -t 1'));
gaussian = svmtrain(y_train, x_train, sprintf('-s 0 -t 2'));
sigmoid = svmtrain(y_train, x_train, sprintf('-s 0 -t 3'));

[predict_label_linear_train, training_accuracy_linear, dec_values_linear_train] = svmpredict(y_train, x_train, linear);
[predict_label_poly_train, training_accuracy_poly, dec_values_poly_train] = svmpredict(y_train, x_train, poly);
[predict_label_gaussian_train, training_accuracy_gaussian, dec_values_gaussian_train] = svmpredict(y_train, x_train, gaussian);
[predict_label_sigmoid_train, training_accuracy_sigmoid, dec_values_sigmoid_train] = svmpredict(y_train, x_train, sigmoid);

[predict_label_linear_test, testing_accuracy_linear, dec_values_linear_test] = svmpredict(y_test, x_test, linear);
[predict_label_poly_test, testing_accuracy_poly, dec_values_poly_test] = svmpredict(y_test, x_test, poly);
[predict_label_gaussian_test, testing_accuracy_gaussian, dec_values_gaussian_test] = svmpredict(y_test, x_test, gaussian);
[predict_label_sigmoid_test, testing_accuracy_sigmoid, dec_values_sigmoid_test] = svmpredict(y_test, x_test, sigmoid);

% -------------------------------------------------------------------------

% c:-----------------------------------------------------------------------
gamma1 = 10000;
C1 = 1;
gamma2 = 10;
C2 = 2000;

classifier1 = svmtrain(y_train, x_train, sprintf('-s 0 -t 2'));
classifier2 = svmtrain(y_train, x_train, sprintf('-s 0 -t 2'));

