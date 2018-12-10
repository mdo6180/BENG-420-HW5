clear
close all

data = readtable('CarcinomaNormalDataset.xls','Sheet','Cancer','Range','A1:AM13');
data = removevars(data,{'Sample'});
data(9,:) = [];
data(end,:) = [];
data = data(:,[3:end]);
data = table2array(data(:,:));
data = data';

