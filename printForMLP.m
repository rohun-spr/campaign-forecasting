function printForMLP(size, lr, momentum, YTRAIN, YFITTRAIN, xTrainLength, YTEST, YTESTFIT, xTestLength, errorOutFile, TAG)

rSquaredTrain = justRSquaredError(YTRAIN', YFITTRAIN');
mseTrain = justMSE(YTRAIN',YFITTRAIN');

scatter(xTrainLength,YTRAIN,'g')
hold on
scatter(xTrainLength,YFITTRAIN','b')
train_image_name = strcat('images/Train_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.png');
saveas(gcf, train_image_name);
hold off

rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
mseTest = justMSE(YTEST,YTESTFIT');

scatter(xTestLength,YTEST','g')
hold on
scatter(xTestLength,YTESTFIT','b')
test_image_name = strcat('images/Test_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.png');
saveas(gcf, test_image_name);
hold off

prediction_file_name = strcat('predictions/model_predictions_on_TRAIN_size', num2str(size), '_mc_', num2str(momentum), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.csv');
fid=fopen(prediction_file_name,'w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTRAIN)
    actual = YTRAIN(idx);
    prediction = YFITTRAIN(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end

prediction_file_name = strcat('predictions/model_predictions_on_TEST_size', num2str(size), '_mc_', num2str(momentum), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.csv');
fid=fopen(prediction_file_name,'w');
fprintf(fid, 'actual,prediction\n');
for idx = 1:length(YTEST)
    actual = YTEST(idx);
    prediction = YTESTFIT(idx);
    fprintf(fid, '%f,%f\n', actual, prediction);
end

fprintf(errorOutFile, '%f,%f,%f,%f,%f,%f,%f\n', lr, momentum, size, rSquaredTrain, mseTrain, rSquaredTest, mseTest);
end