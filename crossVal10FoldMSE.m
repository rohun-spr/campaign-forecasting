train_data = csvread('outputFeatures.csv');

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 11;
PREDICTION_INDEX = 12;

XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTRAIN = train_data(:,PREDICTION_INDEX);

predFuncHandle = @(XTRAIN, YTRAIN,XTEST)(predFunc(XTRAIN, YTRAIN, XTEST));

cvMse = crossval('mse',XTRAIN, YTRAIN,'Predfun',predFuncHandle);
cvMse