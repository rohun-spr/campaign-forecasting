data_matrix = csvread('zoo.csv');

X = data_matrix(:,1:16);
Y = data_matrix(:, 17);

c = cvpartition(Y,'k',10);
opts = statset('display','iter');

r_squared_error_handle = @(XT,yT,Xt,yt)(rSquaredError(XT, yT, Xt, yt));
%function error = rSquaredError(XTRAIN, YTRAIN, XTEST, YTEST)

inmodel = sequentialfs(r_squared_error_handle,X,Y,'cv',c,'options',opts);
inmodel