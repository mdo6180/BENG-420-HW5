clear
close all

% Load training features and labels
[y, x] = libsvmread('data_smallest.txt');
%[y, x] = libsvmread('data_largest.txt');

% Libsvm options
% -s: 0 = multi-class classification
% -t: 0 = linear kernel, 1 = polynomial, 2 = Gaussian, 3 = sigmoid 

% leave-1-out cross validation
avg_sse1 = 0;
for i = 1:length(y)
    leave_out_x = x(i,:);
    leave_out_y = y(i);
    
    x(i,:) = [];
    y(i) = [];
    
    % uncomment lines 22 and 23 to run linear model
%     linear_model = svmtrain(y, x, sprintf('-s 0 -t 0'));    % train linear classifier
%     [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);

    % uncomment lines 26 and 27 to run gaussian model
    gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2'));    % train gaussian classifier
    [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, gaussian_model);
    
    sse1 = (leave_out_y - predict_label)^2;
    avg_sse1 = avg_sse1 + sse1;
    
    x = [leave_out_x; x];
    y = [leave_out_y; y];
end
avg_sse1 = avg_sse1/length(y)   % average sum-square-error for all runs

% leave-10-out cross validation
avg_sse10 = 0;    % sum of mean square error every run
for i = 1:(length(y) - 10)
    leave_out_x = x(i:(i + 9),:);
    leave_out_y = y(i:(i + 9));
    
    x(i:(i + 9),:) = [];
    y(i:(i + 9)) = [];
    
    % uncomment lines 47 and 48 to run linear model
%     linear_model = svmtrain(y, x, sprintf('-s 0 -t 0'));    % train linear classifier
%     [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);

    % uncomment lines 51 and 52 to run gaussian model
    gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2'));  % train gaussian classifier
    [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, gaussian_model);
    
    sse10 = sum((leave_out_y - predict_label).^2);   % sum-squared-error
    
    avg_sse10 = avg_sse10 + sse10;  
    
    x = [leave_out_x; x];
    y = [leave_out_y; y];
end
avg_sse10 = avg_sse10/length(y)     % average sum-squared-error for all runs
