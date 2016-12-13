function usingGlmFit()

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 3;
PREDICTION_INDEX = FEATURE_STOP_INDEX + 1;
TAG = 'LINEAR_TW_CATEGORICAL';

train_data = csvread('outputFeaturesTrain.csv');
XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTRAIN = train_data(:,PREDICTION_INDEX);

test_data = csvread('outputFeaturesTest.csv');
XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
YTEST = test_data(:,PREDICTION_INDEX);

link = @(mu) log(mu ./ (1-mu));
derlink = @(mu) 1 ./ (mu .* (1-mu));
invlink = @(resp) 1 ./ (1 + exp(-resp));
F = {link, derlink, invlink};

b = glmfit(XTRAIN,YTRAIN,'binomial','link',F);

yfitTrain = glmval(b,XTRAIN,F,'size',2);


X = 1:length(XTRAIN);
plot(X,YTRAIN,'--go',X,yfitTrain',':r*')
    
fid=fopen('model_predictions_on_training.csv','w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTRAIN)
    actual = YTRAIN(idx);
    prediction = yfitTrain(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end
%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%

yfitTest = glmval(b,XTEST,F,'size',2);

X = 1:length(XTEST);
plot(X,YTEST,'--go',X,yfitTest',':r*')

fid=fopen('model_predictions_on_test_set.csv','w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTEST)
    actual = YTEST(idx);
    prediction = yfitTest(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end
end