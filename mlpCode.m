function mlpCode()

TAG = 'TW_FULL_DATA_5_DIFFERENT_MOMENTUMS_FOR_CLUSTERS_CATEGORICAL_NO_CPE';

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 12;
PREDICTION_INDEX = FEATURE_STOP_INDEX + 1;

CLUSTER_COUNT = 5;
TRAIN_TO_TOTAL = 0.7;

% test_data = csvread('outputFeaturesTest.csv');
% XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
% YTEST = test_data(:,PREDICTION_INDEX)';

train_data = csvread('outputFeatures.csv');
DATA_TOTAL = train_data(:,FEATURE_START_INDEX:PREDICTION_INDEX);

model_error_file_name = strcat('resultant_error/model_error_', TAG, '.csv');
errorOutFile=fopen(model_error_file_name,'w');
fprintf(errorOutFile, 'Learning-Rate,Momentum,Cluster-Number,TrainR-Squared,Train-MSE,TestR-Squared,Test-MSE\n');

rng(1)
idx = kmeans(DATA_TOTAL(:,FEATURE_START_INDEX-1:FEATURE_STOP_INDEX-1),CLUSTER_COUNT);
%idx = kmeans(DATA_TOTAL,CLUSTER_COUNT);

momentum_list = [0.7, 0.7, 0.55, 0.7, 0.7];

size = 5; % Default Hidden Units
for clusterNumber = 1:CLUSTER_COUNT
    CLUSTER_DATA = DATA_TOTAL(idx==clusterNumber,FEATURE_START_INDEX-1:PREDICTION_INDEX-1);
    
    train_portion = round(length(CLUSTER_DATA) * TRAIN_TO_TOTAL);
    XTRAIN = CLUSTER_DATA(1:train_portion,FEATURE_START_INDEX-1:FEATURE_STOP_INDEX-1)';
    YTRAIN = CLUSTER_DATA(1:train_portion,PREDICTION_INDEX-1)';
    
    XTEST = CLUSTER_DATA(train_portion:end,FEATURE_START_INDEX-1:FEATURE_STOP_INDEX-1)';
    YTEST = CLUSTER_DATA(train_portion:end,PREDICTION_INDEX-1)';

    for lr = [0.05]
        momentum = momentum_list(clusterNumber);
        net = fitnet(size,'traincgb');
        net.layers{1}.transferFcn='logsig';
        net.layers{2}.transferFcn='purelin';
        net.trainParam.lr = lr;
        net.trainParam.mc=momentum;
        
        net.trainParam.epochs = 2000;
        net.performFcn = 'mse';
        net = train(net,XTRAIN,YTRAIN);
        
        YFITTRAIN = net(XTRAIN);
        YTESTFIT = net(XTEST);
        
        printForMLP(clusterNumber, lr, momentum, YTRAIN, YFITTRAIN, 1:length(XTRAIN), YTEST, YTESTFIT, 1:length(XTEST), errorOutFile, TAG)
        
    end
end

end