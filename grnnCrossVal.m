FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 7;
PREDICTION_INDEX = 8;

train_data = csvread('outputFeaturesTrain.csv');
XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTRAIN = train_data(:,PREDICTION_INDEX);

test_data = csvread('outputFeaturesTest.csv');
XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTEST = test_data(:,PREDICTION_INDEX);

fidOutFile=fopen('result_output_file_previous.csv','w');
fprintf(fidOutFile, 'Spread,trainRSquared,trainMSE,testRSquared,testMSE\n');


for spread = [0.2, 0.4, 0.6, 0.8]
    
    net = newgrnn(XTRAIN', YTRAIN', spread);
    YFITTRAIN = net(XTRAIN');
    rSquaredTrain = justRSquaredError(YTRAIN, YFITTRAIN');
    mseTrain = justMSE(YTRAIN, YFITTRAIN');
    rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
    mseTest = justMSE(YTEST, YTESTFIT');

    
    fprintf(fidOutFile, '%f,%f,%f,%f,%f\n', spread, rSquaredTrain, mseTrain, rSquaredTest, mseTest);

end