grnn_data = csvread('weighted_mean/testGRNN.csv');

GRNN_WEIGHT = 0.3;

YTEST = grnn_data(:,1);
YTESTFIT_GRNN = grnn_data(:,2);

mlp_data = csvread('weighted_mean/testMLP.csv');
YESTFIT_MLP = mlp_data(:,2);

z = cat(2,YESTFIT_MLP,YTESTFIT_GRNN);
k = [1 - GRNN_WEIGHT, GRNN_WEIGHT]*z';
YTESTFIT = k';

TAG = strcat('WEIGHTED_RESULT_GRNN_WEIGHT_', num2str(GRNN_WEIGHT));

model_error_file_name = strcat('resultant_error/model_error_', TAG, '.csv');
errorOutFile=fopen(model_error_file_name,'w');
fprintf(errorOutFile, 'Learning-Rate,Momentum,Hidden-Units,TrainR-Squared,Train-MSE,TestR-Squared,Test-MSE\n');

printForMLP(1, 1, 1, YTEST, YTESTFIT, 1:length(YTEST), YTEST, YTESTFIT, 1:length(YTEST), errorOutFile, TAG)