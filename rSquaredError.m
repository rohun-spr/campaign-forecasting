function error = rSquaredError(XTRAIN, YTRAIN, XTEST, YTEST)
%This error value is calculated by fitting the model on the train data
% and reporting the rsquared error on the test set.

YTESTFIT = predFunc(XTRAIN, YTRAIN, XTEST);

sum_init = 0;
for index = 1:length(YTEST)
    sum_init = sum_init + YTEST(index);
end
mean = sum_init/length(YTEST);

numerator = 0;
denominator = 0;

for index = 1:length(YTEST)
    prediction = YTESTFIT(index);
    actual = YTEST(index);
    num = prediction-mean;
    denom = actual-mean;
    numerator = numerator + num * num;
    denominator = denominator + denom * denom;
end

X = 1:length(YTEST);

figure
axis([-inf,inf,0,0.010])
plot(X,YTEST,'--go',X,YTESTFIT,':r*')

error = numerator/denominator;
error