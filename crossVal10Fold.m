train_data = csvread('outputFeatures.csv');

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 11;
PREDICTION_INDEX = 12;

XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTRAIN = train_data(:,PREDICTION_INDEX);

r_squared_error_handle = @(XTRAIN, YTRAIN, XTEST, YTEST)(rSquaredError(XTRAIN, YTRAIN, XTEST, YTEST));

cvMse = crossval(r_squared_error_handle,XTRAIN, YTRAIN);
cvMse
sum(cvMse)/length(cvMse)