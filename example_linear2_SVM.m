% SVM Linear classification
% A 2-feature example
close all
clear all;
figure;

% Load training features and labels
[y, x] = libsvmread('twofeatures.txt');

% remove outlier point
%  y = y(1:50);
%  x = x(1:50,:);

% y=[y; 1];
% x=[x;2 3];

% Plot the data points
pos = find(y == 1);
neg = find(y == -1);
plot(x(pos,1), x(pos,2), 'ko', 'MarkerFaceColor', 'b'); hold on;
plot(x(neg,1), x(neg,2), 'ko', 'MarkerFaceColor', 'g')

% Set the cost
C = 0.1; %30 vs 10000 vs 5 vs 0.02

% Train the model and get the primal variables w, b from the model
% Libsvm options
% -s 0 : classification
% -t 0 : linear kernel
% -c some number : set the cost
model = svmtrain(y, x, sprintf('-s 0 -t 0 -c %g', C));
theta = model.SVs' * model.sv_coef
theta0 = -model.rho
theta0 = (1 - theta'*model.SVs(1,:)' + 1 - theta'*model.SVs(2,:)' + (-1) - theta'*model.SVs(3,:)') / 3

% Plot the decision boundary
plot_x = linspace(min(x(:,1)), max(x(:,1)), 30);
plot_y = (-1/theta(2)) * (theta(1)*plot_x + theta0);
plot(plot_x, plot_y, 'k:', 'LineWidth', 2)

% Plot the margin boundaries
plot_y = (-1/theta(2)) * (theta(1)*plot_x + theta0 + 1);
plot(plot_x, plot_y, 'c:', 'LineWidth', 2);
plot_y = (-1/theta(2)) * (theta(1)*plot_x + theta0 - 1);
plot(plot_x, plot_y, 'c:', 'LineWidth', 2)

% identify support vectors
for i=1:size(model.SVs, 1)
    plot(model.SVs(i, 1), model.SVs(i, 2), 'ro', 'MarkerSize', 10);
end

title(sprintf('SVM Linear Classifier with C = %g', C), 'FontSize', 14)

[predicted_label, accuracy, decision_values] = svmpredict(y, x, model);
accuracy

x_p = [2 3.4];
svmpredict(-1, x_p, model)
plot(x_p(1), x_p(2), 'pm', 'MarkerSize', 10);

x_p = [4 2.6];
svmpredict(1, x_p, model)
plot(x_p(1), x_p(2), 'pk', 'MarkerSize', 10);