train_data = csvread('testR.csv');
XTRAIN = train_data(:,1);
YTRAIN = train_data(:,2);

error = justRSquaredError(XTRAIN,YTRAIN);
error