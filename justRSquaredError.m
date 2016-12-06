function error = justRSquaredError(YTEST,YTESTFIT)
%This error value is calculated by fitting the model on the train data
% and reporting the rsquared error on the test set.



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
    num = actual-prediction;
    denom = actual-mean;
    numerator = numerator + num * num;
    denominator = denominator + denom * denom;
end

error = 1 - numerator/denominator;