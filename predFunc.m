function [yfitTest] = predFunc(XTRAIN, YTRAIN, XTEST)
%This function calculates the yfit Value for the test set after
% training the glm model using the train set

link = @(mu) log(mu ./ (1-mu));
derlink = @(mu) 1 ./ (mu .* (1-mu));
invlink = @(resp) 1 ./ (1 + exp(-resp));
F = {link, derlink, invlink};

b = glmfit(XTRAIN,YTRAIN,'binomial','link',F);
%%%%%%%%%%%%%%%%%%%%%%TESTING%%%%%%%%%%%%%%%%%%%%%

yfitTest = glmval(b,XTEST,F,'size',2);

