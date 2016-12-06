function mlpCode()
FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 13;
PREDICTION_INDEX = FEATURE_STOP_INDEX + 1;
TAG = 'FEED_FORWARD_NO_COUNTRY_ONLY_PROB';

train_data = csvread('outputFeaturesTrain.csv');
XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTRAIN = train_data(:,PREDICTION_INDEX)';

test_data = csvread('outputFeaturesTest.csv');
XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTEST = test_data(:,PREDICTION_INDEX)';

model_error_file_name = strcat('model_error_', TAG, '.csv');
errorOutFile=fopen(model_error_file_name,'w');
fprintf(errorOutFile, 'TrainR-Squared,TrainMSE,TestR-Squared,TestMSE\n');

%Train the model
net = feedforwardnet([10, 10],'trainscg');
net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='logsig';
net.layers{3}.transferFcn='purelin';
net.trainParam.lr = 0.1;

net.trainParam.epochs = 1000;
net.performFcn = 'mse';
net = train(net,XTRAIN,YTRAIN);

%get the prediction
YFITTRAIN = net(XTRAIN);

rSquaredTrain = justRSquaredError(YTRAIN', YFITTRAIN');
mseTrain = perform(net,YTRAIN',YFITTRAIN');

X = 1:length(XTRAIN);
plot(X,YTRAIN,'--go',X,YFITTRAIN',':r*')
%train_image_name = strcat('images/Train_Plot_Spread_', num2str(spread), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), '.png');
%saveas(gcf, train_image_name);
%%%%%%%%%%%%%%%%%%%%%%%%

YTESTFIT = net(XTEST);

rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
mseTest = perform(net,YTEST,YTESTFIT');

X = 1:length(XTEST);
plot(X,YTEST,'--go',X,YTESTFIT',':r*')
%test_image_name = strcat('images/Test_Plot_Spread_', num2str(spread), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), '.png');
%saveas(gcf, test_image_name);

prediction_file_name = strcat('predictions/model_predictions_on_TRAIN', '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.csv');
fid=fopen(prediction_file_name,'w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTRAIN)
    actual = YTRAIN(idx);
    prediction = YFITTRAIN(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end

prediction_file_name = strcat('predictions/model_predictions_on_TEST', '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.csv');
fid=fopen(prediction_file_name,'w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTEST)
    actual = YTEST(idx);
    prediction = YTESTFIT(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end

fprintf(errorOutFile, '%f,%f,%f,%f\n', rSquaredTrain, mseTrain, rSquaredTest, mseTest);
end