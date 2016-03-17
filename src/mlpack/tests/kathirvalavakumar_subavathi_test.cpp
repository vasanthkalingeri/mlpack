/*
  Tests the working of kathirvalavakumar_subavathi_init.cpp
  Checks whether the weight_initialization done by the method works as intended.
*/
#include <iostream>
#include <fstream>

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>


#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(KathirvalavakumarSubavathiInitializationTest);

/**
 * Train and evaluate a vanilla network with the specified structure.
 */
template<
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = mat
>
void BuildVanillaNetwork(MatType& trainData,
                         MatType& trainLabels,
                         MatType& testData,
                         MatType& testLabels,
                         const size_t hiddenLayerSize,
                         const size_t maxEpochs,
                         double* mse_errors)
{
  LinearLayer<> inputLayer(trainData.n_rows, hiddenLayerSize);
  BiasLayer<> inputBiasLayer(hiddenLayerSize);
  BaseLayer<PerformanceFunction> inputBaseLayer;

  LinearLayer<> hiddenLayer1(hiddenLayerSize, trainLabels.n_rows);
  BiasLayer<> hiddenBiasLayer1(trainLabels.n_rows);
  BaseLayer<PerformanceFunction> outputLayer;
  
  OutputLayerType classOutputLayer;

  auto modules = std::tie(inputLayer, inputBiasLayer, inputBaseLayer,
                          hiddenLayer1, hiddenBiasLayer1, outputLayer);

  KathirvalavakumarSubavathiInitialization weight_initialization(trainData, 4.59); //Constant used in paper
  FFN<decltype(modules), decltype(classOutputLayer), decltype(weight_initialization),
      PerformanceFunctionType> net(modules, classOutputLayer, weight_initialization);

  RMSprop<decltype(net)> opt(net, 0.01, 0.88, 1e-8,
      maxEpochs * trainData.n_cols, 1e-18);

  net.Train(trainData, trainLabels, opt);
  
  MatType prediction;
  
  net.Predict(testData, prediction);
  //Find the Mean square error
  mat errors = pow((prediction*1.0 - testLabels), 2);
  mse_errors[0] = sum(sum(errors)) * 1.0 / testData.n_cols;
  
  //Repeat for training
  net.Predict(trainData, prediction);
  errors = pow((prediction*1.0 - trainLabels), 2);
  mse_errors[1] = sum(sum(errors)) * 1.0 / trainData.n_cols;
  
  return;
}

void analyze(mat& Data,
            mat& Labels, 
            const size_t num_folds, 
            const size_t hiddenLayerSize, 
            const size_t epochs, 
            double *errors)
{
  /*
    The function shuffles the data and performs K-fold cross validation. It reports the average of the error obtained accross all folds to be the training/validation error.
    
    @params Data The data used to train the network
    @params Labels The labels for the training data
    @params num_folds the parameter 'K' to be used for K-fold validation
    @params hiddenLayerSize The number of nodes to be present in the hidden layer
    @params epochs Net parameter
    @params errors The training and CV error obtained on the data
  */
  
  //Train and validation accuracy
  double train_acc = 0, cv_acc = 0;
  
  //Shuffled indices
  uvec indices = shuffle(linspace<uvec>(0, Data.n_cols - 1, Data.n_cols));
  mat shuffledData(Data.n_rows, Data.n_cols);
  mat shuffledLabels(Labels.n_rows, Labels.n_cols);
  for (size_t i = 0; i < Data.n_cols; ++i)
  {
    shuffledData.col(i) = Data.col(indices[i]);
    shuffledLabels.col(i) = Labels.col(indices[i]);
  }

  
  //Obtain the number of elements in each fold
  size_t no_elements = (Data.n_cols / num_folds);
  
  //Have to take all rows of the dataset [Take all the features of the data]
  uvec data_rows = linspace<uvec>(0, Data.n_rows - 1, shuffledData.n_rows);
  uvec label_rows = linspace<uvec>(0, Labels.n_rows - 1, shuffledLabels.n_rows);
  uvec val_cols, data_cols, top, bottom;
  for(size_t i = 0; i < num_folds; ++i)
  {
    if (i == (num_folds - 1))
    {
      //Take from start till last but one fold for training
      data_cols = linspace<uvec>(0, no_elements*i - 1, no_elements*i);
      
      //Take remaining elements to be validation
      val_cols = linspace<uvec>(no_elements*i, shuffledData.n_cols - 1, shuffledData.n_cols - no_elements*i);
    }
    else
    {
      //Choose the fold for validation
      val_cols = linspace<uvec>(no_elements*i, (i + 1)*no_elements - 1, no_elements);

      //Choose remaining for training
      if(i > 0)
      {
        top = linspace<uvec>(0, no_elements*i - 1, no_elements*i);
        bottom = linspace<uvec>((i+1)*no_elements, shuffledData.n_cols - 1, (shuffledData.n_cols - no_elements*(i + 1)));
        data_cols = join_cols(top, bottom);
      }
      else
        data_cols = linspace<uvec>((i+1)*no_elements, shuffledData.n_cols - 1, (shuffledData.n_cols - no_elements*(i + 1)));
    }
  
    mat trainData = shuffledData.submat(data_rows, data_cols);
    mat trainLabels =  shuffledLabels.submat(label_rows, data_cols);
    mat valData = shuffledData.submat(data_rows, val_cols);
    mat valLabels = shuffledLabels.submat(label_rows, val_cols);
    double mse_errors[2] = {0.0, 0.0};
    BuildVanillaNetwork<LogisticFunction, BinaryClassificationLayer, MeanSquaredErrorFunction>
      (trainData, trainLabels, valData, valLabels, hiddenLayerSize, epochs, mse_errors);
    
    cv_acc += mse_errors[0];
    train_acc += mse_errors[1]; 
    mse_errors[0] = 0;
    mse_errors[1] = 0; 
  }  
  errors[0] = cv_acc / num_folds;
  errors[1] = train_acc / num_folds;
  return;
}

void run(mat& Data,
        mat& Labels, 
        const size_t hiddenLayerSize, 
        const size_t epochs, 
        const size_t train_threshold, 
        const size_t cv_threshold,
        const size_t num_folds)
{
  /*
    Runs the entire network multiple times and returns the average cross validation and training error obtained accross all the runs.
    @params Data The data used to train the network
    @params Labels The labels for the training data
    @params num_folds the parameter 'K' to be used for K-fold validation
    @params hiddenLayerSize The number of nodes to be present in the hidden layer
    @params epochs Net parameter
    @params train_threshold the error limits for training
    @params cv_threshold the error limits for validation
    @params errors The training and CV error obtained on the data
  */
  
  double errors[2] = {0.0, 0.0};
  const size_t num_tries = 5;
  double train_error = 0, cv_error = 0;
  for(size_t i = 0; i < num_tries; ++i)
  {
    analyze(Data, Labels, num_folds, hiddenLayerSize, epochs, errors);
    cv_error += errors[0];
    train_error += errors[1];
  }
  train_error /= num_tries;
  cv_error /= num_tries;
  BOOST_REQUIRE_LE(train_error, train_threshold);
  BOOST_REQUIRE_LE(cv_error, cv_threshold);
  return;
}

BOOST_AUTO_TEST_CASE(IrisDataTest)
{
  mat Data, Labels;
  data::Load("iris.csv", Data);
  data::Load("iris_labels.txt", Labels);
  size_t hiddenLayerSize, epochs, num_folds;
  
  //Testing the 5-3-1 network with thresholds 0.4 for both
  hiddenLayerSize = 2;
  epochs = 61;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.4, 0.4, num_folds);
  
  //Testing the 5-7-1 network with thresholds 0.4 for both
  hiddenLayerSize = 6;
  epochs = 61;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.4, 0.4, num_folds);
  
  //Testing the 5-9-1 network with thresholds 0.4 for both
  hiddenLayerSize = 8;
  epochs = 89;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.4, 0.4, num_folds);
  
  //Testing the 5-10-1 network with thresholds 0.4 for both
  hiddenLayerSize = 9;
  epochs = 23;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.4, 0.4, num_folds);
}

BOOST_AUTO_TEST_CASE(TwoSpiralsTest)
{
  mat Data, Labels;
  data::Load("spirals.csv", Data);
  data::Load("spirals_labels.txt", Labels);
  size_t hiddenLayerSize, epochs, num_folds;
  
  //Testing the 3-8-1 network with thresholds 0.7 for both
  hiddenLayerSize = 7;
  epochs = 212;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.7, 0.7, num_folds);
  
  //Testing the 3-10-1 network with thresholds 0.7 for both
  hiddenLayerSize = 9;
  epochs = 156;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.7, 0.7, num_folds);
  
  //Testing the 3-11-1 network with thresholds 0.7 for both
  hiddenLayerSize = 10;
  epochs = 194;
  num_folds = 10;
  run(Data, Labels, hiddenLayerSize, epochs, 0.7, 0.7, num_folds);
}

BOOST_AUTO_TEST_SUITE_END();
