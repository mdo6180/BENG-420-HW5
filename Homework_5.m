clear
close all

% Load training features and labels
[y, x] = libsvmread('data.txt');

% Libsvm options
% -s: 0 = classification
% -t: 2 = Gaussian kernel, 0 = linear kernel
% -g : gamma in the Gaussian kernel
% -c : cost

% Set the cost
C = 30; %30 vs 10000 vs 5 vs 0.02
linear_model = svmtrain(y, x, sprintf('-s 0 -t 0 -c %g', C));

% Set gamma
gamma = 10000;
gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma));

% leave-1-out cross validation
% mse1 = 0;    % sum of mean square error every run
% for i = 1:length(y)
%     leave_out_x = x(i,:);
%     leave_out_y = y(i);
%     
%     x(i,:) = [];
%     y(i) = [];
%     
%     %cv_linear_model = svmtrain(y, x, sprintf('-s 0 -t 0 -c %g', C));
%     cv_gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma));
%     [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);
%     
%     error = (leave_out_y - predict_label)^2;
%     %mse1 = mse1 + accuracy(2);
%     mse1 = mse1 + error;
%     
%     x = [leave_out_x; x];
%     y = [leave_out_y; y];
% end
% mse1 = 100*(mse1/length(y))

% leave-10-out cross validation
mse10 = 0;    % sum of mean square error every run
for i = 1:(length(y) - 10)
    leave_out_x = x(i:(i + 9),:);
    leave_out_y = y(i:(i + 9));
    
    x(i:(i + 9),:) = [];
    y(i:(i + 9)) = [];
    
    cv_gaussian_model = svmtrain(y, x, sprintf('-s 0 -t 2 -g %g', gamma));
    [predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);
    %cv_linear_model = svmtrain(y, x, sprintf('-s 0 -t 0 -c %g', C));
    %[predict_label, accuracy, dec_values] = svmpredict(leave_out_y, leave_out_x, linear_model);
    
    error = sum((leave_out_y - predict_label).^2);
    %mse10 = mse10 + accuracy(2);
    mse10 = mse10 + error;
    
    x = [leave_out_x; x];
    y = [leave_out_y; y];
end
mse10 = 100*(mse10/length(y))