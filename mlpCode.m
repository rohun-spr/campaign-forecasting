function mlpCode()

TAG = 'TWITTER_ONLY_PROB_0.3_PORTION_LOGSIG_PURELIN';

FEATURE_START_INDEX = 2;
FEATURE_STOP_INDEX = 12;
PREDICTION_INDEX = FEATURE_STOP_INDEX + 1;

train_data = csvread('outputFeaturesTrain.csv');
XTRAIN = train_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTRAIN = train_data(:,PREDICTION_INDEX)';

test_data = csvread('outputFeaturesTest.csv');
XTEST = test_data(:,FEATURE_START_INDEX:FEATURE_STOP_INDEX)';
YTEST = test_data(:,PREDICTION_INDEX)';

model_error_file_name = strcat('resultant_error/model_error_', TAG, '.csv');
errorOutFile=fopen(model_error_file_name,'w');
fprintf(errorOutFile, 'Learning-Rate,Momentum,Hidden-Units,TrainR-Squared,Train-MSE,TestR-Squared,Test-MSE\n');

% Hidden Layer
size = 5; % Default Hidden Units
for lr = [0.05]
    for momentum = [0.6, 0.65, 0.7]
        net = fitnet(5,'traingdm');
        net.layers{1}.transferFcn='logsig';
        net.layers{1}.transferFcn='purelin';
        net.trainParam.lr = lr;
        net.trainParam.mc=momentum;

        net.trainParam.epochs = 2000;
        net.performFcn = 'mse';
        net = train(net,XTRAIN,YTRAIN);

        %get the prediction
        YFITTRAIN = net(XTRAIN);
        
        rSquaredTrain = justRSquaredError(YTRAIN', YFITTRAIN');
        mseTrain = perform(net,YTRAIN',YFITTRAIN');

        X = 1:length(XTRAIN);
        plot(X,YTRAIN,'--go',X,YFITTRAIN',':r*')
        train_image_name = strcat('images/Train_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.png');
        saveas(gcf, train_image_name);
        %%%%%%%%%%%%%%%%%%%%%%%%

        YTESTFIT = net(XTEST);

        if momentum == 0.6
            YFITTRAIN_six = YFITTRAIN;
            YFITTEST_six = YTESTFIT;
        end
        if momentum == 0.7
            YFITTRAIN_seven = YFITTRAIN;
            YFITTEST_seven = YTESTFIT;
        end
        
        rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
        mseTest = perform(net,YTEST,YTESTFIT');

        X = 1:length(XTEST);
        plot(X,YTEST,'--go',X,YTESTFIT',':r*')
        test_image_name = strcat('images/Test_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.png');
        saveas(gcf, test_image_name);

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
    
    momentum = 0.0607;
    
    z = cat(3,YFITTRAIN_six,YFITTRAIN_seven);
    YFITTRAIN = mean(z,3);
    
    z = cat(3,YFITTEST_six,YFITTEST_seven);
    YTESTFIT = mean(z,3);
    
    rSquaredTrain = justRSquaredError(YTRAIN', YFITTRAIN');
    mseTrain = perform(net,YTRAIN',YFITTRAIN');

    X = 1:length(XTRAIN);
    plot(X,YTRAIN,'--go',X,YFITTRAIN',':r*')
    train_image_name = strcat('images/Train_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTrain), '_mse_', num2str(mseTrain), TAG, '.png');
    saveas(gcf, train_image_name);

    rSquaredTest = justRSquaredError(YTEST, YTESTFIT');
    mseTest = perform(net,YTEST,YTESTFIT');

    X = 1:length(XTEST);
    plot(X,YTEST,'--go',X,YTESTFIT',':r*')
    test_image_name = strcat('images/Test_Plot_size', num2str(size), '_lr_', num2str(lr), '_mc_', num2str(momentum), '_rsquared_', num2str(rSquaredTest), '_mse_', num2str(mseTest), TAG, '.png');
    saveas(gcf, test_image_name);

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

end