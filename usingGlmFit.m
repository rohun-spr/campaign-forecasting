train_data = csvread('outputFeaturesTrain.csv');

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 11;
PREDICTION_INDEX = 12;

X = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
Y = train_data(:,PREDICTION_INDEX);

link = @(mu) log(mu ./ (1-mu));
derlink = @(mu) 1 ./ (mu .* (1-mu));
invlink = @(resp) 1 ./ (1 + exp(-resp));
F = {link, derlink, invlink};

b = glmfit(X,Y,'binomial','link',F);

yfitTrain = glmval(b,X,F,'size',2);
fid=fopen('model_predictions_on_training.csv','w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(Y)
    actual = Y(idx);
    prediction = yfitTrain(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end
%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%

test_data = csvread('outputFeaturesTest.csv');
X = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX);
Y = test_data(:,PREDICTION_INDEX);

yfitTest = glmval(b,X,F,'size',2);

fid=fopen('model_predictions_on_test_set.csv','w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(Y)
    actual = Y(idx);
    prediction = yfitTest(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end