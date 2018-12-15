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

classifier1 = svmtrain(y_train, x_train, sprintf('-s 0 -t 2 -g %d -c %d', gamma1, C1));
classifier2 = svmtrain(y_train, x_train, sprintf('-s 0 -t 2 -g %d -c %d', gamma2, C2));

% Compute training accuracy (lines 41 and 42)
% [predict_label1, accuracy1, dec_values1] = svmpredict(y_train, x_train, classifier1);     
% [predict_label2, accuracy2, dec_values2] = svmpredict(y_train, x_train, classifier2);

% Compute testing accuracy (lines 45 and 46)
[predict_label1, accuracy1, dec_values1] = svmpredict(y_test, x_test, classifier1);
[predict_label2, accuracy2, dec_values2] = svmpredict(y_test, x_test, classifier2);
% -------------------------------------------------------------------------

% d:-----------------------------------------------------------------------
optimal_gamma = 0;
optimal_C = 0;
highest_accuracy = 0;
for i = 0.1:0.5:10      % range of gamma
    for j = 0.1:0.1:2   % range of C
        
        model = svmtrain(y_train, x_train, sprintf('-s 0 -t 2 -g %d -c %d -h 0', i, j));
        [predict_label_model, accuracy_model, dec_values_model] = svmpredict(y_test, x_test, model);
        
        if accuracy_model(1) > highest_accuracy
            highest_accuracy = accuracy_model(1);
            optimal_gamma = i;
            optimal_C = j;
        end
        
    end
end

model_train = svmtrain(y_train, x_train, sprintf('-s 0 -t 2 -g %d -c %d -h 0', 9.6, 2));
[predict_labels_train, accuracy_train, dec_values_train] = svmpredict(y_train, x_train, model_train);
% -------------------------------------------------------------------------

% e:-----------------------------------------------------------------------
training_features = full(x_train)';
training_labels = y_train';
training_labels(training_labels >= 0) = 1;  % convert all positive class labels to 1
training_labels(training_labels < 0) = 0;   % convert all negative class labels to 0

testing_features = full(x_test)';
testing_labels = y_test';
testing_labels(testing_labels >= 0) = 1;
testing_labels(testing_labels < 0) = 0;

% plot training confusion matrix
net_train = feedforwardnet(4);      % creating feed-forward neural net with 4 hidden neurons
net_train = train(net_train, training_features, training_labels);   % training neural net
predicted_labels_train = net_train(training_features);  % predicting class labels

predicted_labels_train(predicted_labels_train >= 0) = 1;    % convert all positive predicted values to 1
predicted_labels_train(predicted_labels_train < 0) = 0;     % convert all negative predicted values to 1

perf = perform(net_train, predicted_labels_train, training_labels);

figure(1)
plotconfusion(training_labels, predicted_labels_train)      %plot confusion matrix
title('Training Confusion Matrix')

% plot testing confusion matrix
net_test = feedforwardnet(4);
net_test = train(net_test, training_features, training_labels);
predicted_labels_test = net_test(testing_features);

predicted_labels_test(predicted_labels_test >= 0) = 1;
predicted_labels_test(predicted_labels_test < 0) = 0;

perf = perform(net_test, predicted_labels_test, testing_labels);

figure(2)
plotconfusion(testing_labels, predicted_labels_test)
title('Testing Confusion Matrix')
