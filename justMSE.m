function MSE = justMSE(YTEST,YTESTFIT)
%This error value is calculated by comparing the predictions and
%reporting the MSE error on the test set.

D = abs(YTEST-YTESTFIT).^2;
MSE = sum(D(:))/numel(YTEST);