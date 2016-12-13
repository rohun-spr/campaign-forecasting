function twoLayerMlpCode()

TAG = 'FITNET_MZ_TWITTER_TWO_LAYERS_LOGSIG_CATEGORICAL_OUTLIERS_REMOVED';

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 3;
PREDICTION_INDEX = FEATURE_STOP_INDEX + 1;

train_data = csvread('outputFeaturesTrain.csv');
XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTRAIN = train_data(:,PREDICTION_INDEX)';

test_data = csvread('outputFeaturesTest.csv');
XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTEST = test_data(:,PREDICTION_INDEX)';

model_error_file_name = strcat('resultant_error/model_error_', TAG, '.csv');
errorOutFile=fopen(model_error_file_name,'w');
fprintf(errorOutFile, 'Learning-Rate,First-Size,Second-Size,TrainR-Squared,TrainMSE,TestR-Squared,TestMSE\n');
for lr = [0.5, 0.7]
    for firstSize = [80, 80]
        for secondSize = [50, 80]
            
            net = fitnet([firstSize, secondSize],'trainbfg');
            net.layers{1}.transferFcn='logsig';
            net.layers{1}.transferFcn='logsig';

            net.trainParam.lr = lr;
r
            net.trainParam.epochs = 1000;
            net.performFcn = 'mse';
            net = train(net,XTRAIN,YTRAIN);

            %get the prediction
            YFITTRAIN = net(XTRAIN);

            rSquaredTrain = justRSquaredError(YTRAIN', YFITTRAIN');
            mseTrain = perform(net,YTRAIN',YFITTRAIN');

            X = 1:length(XTRAIN);
            plot(X,YTRAIN,'--go',X,YFITTRAIN',':r*')
            train_image_name = strcat('images/Train_Plot_firstSize', num2str(firstSize), '_secondSize_', num2str(secondSize), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.png');
            saveas(gcf, train_image_name);
            %%%%%%%%%%%%%%%%%%%%%%%%

            YTESTFIT = net(XTEST);

            rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
            mseTest = perform(net,YTEST,YTESTFIT');

            X = 1:length(XTEST);
            plot(X,YTEST,'--go',X,YTESTFIT',':r*')
            test_image_name = strcat('images/Test_Plot_firstSize', num2str(firstSize), '_secondSize_', num2str(secondSize), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.png');
            saveas(gcf, test_image_name);

            prediction_file_name = strcat('predictions/model_predictions_on_TRAIN_firstSize', num2str(firstSize), '_secondSize_', num2str(secondSize), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.csv');
            fid=fopen(prediction_file_name,'w');
            fprintf(fid, 'actual,prediction\n');
            for idx = 1:length(YTRAIN)
                actual = YTRAIN(idx);
                prediction = YFITTRAIN(idx);
                fprintf(fid, '%f,%f\n', actual, prediction);
            end

            prediction_file_name = strcat('predictions/model_predictions_on_TEST_firstSize', num2str(firstSize), '_secondSize_', num2str(secondSize), '_lr_', num2str(lr), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.csv');
            fid=fopen(prediction_file_name,'w');
            fprintf(fid, 'actual,prediction\n');
            for idx = 1:length(YTEST)
                actual = YTEST(idx);
                prediction = YTESTFIT(idx);
                fprintf(fid, '%f,%f\n', actual, prediction);
            end

            fprintf(errorOutFile, '%f,%f,%f,%f,%f,%f,%f\n', lr, firstSize, secondSize, rSquaredTrain, mseTrain, rSquaredTest, mseTest);
        end
    end
end

end