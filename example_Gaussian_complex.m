% SVM Nonlinear classification example to show the influence of Gamma
% Qi Wei

clear all; close all; clc

% Load training features and labels
[y, x] = libsvmread('ex8a.txt');

% overfitting
% larger gamma -> smaller sigma -> narrow kernel
gamma = 10000;

% underfitting
% gamma = 0.2;

% Libsvm options
% -s 0 : classification
% -t 2 : Gaussian kernel
% -g : gamma in the Gaussian kernel

model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma));

% Display training accuracy
[predicted_label, accuracy, decision_values] = svmpredict(y, x, model);

% Plot training data and decision boundary
plotboundary(y, x, model);

title(sprintf('\\gamma = %g', gamma), 'FontSize', 14);