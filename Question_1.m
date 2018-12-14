clear
close all

% Load training features and labels
[y, x] = libsvmread('data_smallest.txt');
%[y, x] = libsvmread('data_largest.txt');

% Libsvm options
% -s: 0 = multi-class classification
% -t: 0 = linear kernel, 1 = polynomial, 2 = Gaussian, 3 = sigmoid 

% leave-1-out cross validation
mse1 = 0;    % sum of mean square error every run
for i = 1:length(y)
    leave_out_x = x(i,:);
    leave_out_y = y(i);
    
    x(i,:) = [];
    y(i) = [];
    
    linear_model = svmtrain(y, x, sprintf('-s 0 -t 0'));    % train linear classifier
    [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);
%     gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2'));    % train gaussian classifier
%     [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, gaussian_model);
    
    error = (leave_out_y - predict_label)^2;
    
    mse1 = mse1 + error;
    
    x = [leave_out_x; x];
    y = [leave_out_y; y];
end
mse1 = 100*(mse1/length(y))

% leave-10-out cross validation
mse10 = 0;    % sum of mean square error every run
for i = 1:(length(y) - 10)
    leave_out_x = x(i:(i + 9),:);
    leave_out_y = y(i:(i + 9));
    
    x(i:(i + 9),:) = [];
    y(i:(i + 9)) = [];
    
    linear_model = svmtrain(y, x, sprintf('-s 0 -t 0'));
    [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);
%     gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2'));
%     [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, gaussian_model);
     
    error = sum((leave_out_y - predict_label).^2);
    
    mse10 = mse10 + error;
    
    x = [leave_out_x; x];
    y = [leave_out_y; y];
end
mse10 = 100*(mse10/length(y))

% [train, test] = crossvalind('LeaveMOut',x,1)