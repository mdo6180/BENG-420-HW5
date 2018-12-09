% BENG420 - SVM with Gaussian kernel
% Visualize the effect of the gamma parameter, the decision boundary, the
% support vectors
% Qi Wei

clear all
close all

n = 30;
rng(1); % For reproducibility
r = sqrt(rand(n,1)); % Radius
t = 2*pi*rand(n,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(n,1)+1); % Radius
t2 = 2*pi*rand(n,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

data3 = [data1;data2];
theclass = ones(2*n,1);
theclass(1:n) = -1;

% y-labels, x-features
y = theclass;
x = data3;

% Libsvm options
% -s 0 : classification
% -t 2 : Gaussian kernel
% -g : gamma in the Gaussian kernel
gamma = 15; %1.5, 0.15, 15
c = 20; %20
model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g -c %g', gamma, c));

% Display training accuracy
[predicted_label, accuracy, decision_values] = svmpredict(y, x, model)

% plot the decision boundary and support vectors
plotboundary(y, x, model);
for i=1:size(model.SVs, 1)
    plot(model.SVs(i, 1), model.SVs(i, 2), 'bp', 'MarkerSize', 12);
end

sigma = sqrt(1/gamma);
project = @(data, sigma) sum(exp(-(squareform( pdist(data, 'euclidean') .^ 2) ./ ( 2*sigma^2)))); 
blue_z = project(data1, sigma)';
red_z = project(data2, sigma)';

allp = project(data3, sigma)';

figure;
hold on;
grid on;

scatter3(data3(1:n,1), data3(1:n,2), allp(1:n), 'r');
scatter3(data3(n+1:end,1), data3(n+1:end,2), allp(n+1:end), 'b');