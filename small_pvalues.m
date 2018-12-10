clear
close all

% read in presorted table
% remove 'Sample column'
% feature 'T87871' appears twice, so I am removing one of them
% removing 'AccessionNumber' and 'Description' column
% convert to array
% transpose matrix so each row is an instance, each column is a feature
data = readtable('CarcinomaNormalDataset.xls','Sheet','Cancer','Range','A1:AM12');
data = removevars(data,{'Sample'});      
data(9,:) = [];     
data = data(:,[3:end]);     
data = table2array(data(:,:));      
features = data';   

% creating class labels matrix, 
% 1 = tumor, 0 = normal
label_vector = zeros(36,1);
label_vector(1:18) = 1;

instance_matrix = sparse(features);     % converting features into sparse matrix
libsvmwrite('data.txt', label_vector, instance_matrix);
