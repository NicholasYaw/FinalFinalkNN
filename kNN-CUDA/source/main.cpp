#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include "CudaAlgorithm.hpp"
#include <iostream>
#include <chrono>

int main()
{
    std::cout << "Starting..." << std::endl << "Reading dataset..." << std::endl;
    constexpr char delimiter = ',';
    auto reader = DataReader{ "../data/dataset.csv", LabelIndex::LAST, delimiter };
    auto data = reader.readDataFlat(10000);

    std::cout << data.getNumberOfRows() << " rows read. " << std::endl << std::endl;

    std::cout << "Performing Min-Max Normalization..." << std::endl;
    const auto duration = runWithTimeMeasurementCpu([&]() {
        MinMaxNormalizer{}.normalize(data);
    });
    std::cout << "Data Normalization Completed: " << duration << " ms" << std::endl << std::endl;

    std::cout << "Splitting Dataset..." << std::endl;
    auto [trainingData, testingData] = splitData(data, 90);
    std::cout << "Training Size : " << trainingData.getNumberOfRows() << " rows" << std::endl;
    std::cout << "Testing Size  : " << testingData.getNumberOfRows() << " rows" << std::endl << std::endl;

    std::cout << "Performing KNN Prediction [CUDA]......" << std::endl;
    const auto predictDuration = runWithTimeMeasurementCpu([&]() {
        Cuda::knn(trainingData, testingData);
    });

    std::cout << "Prediction Completed: " << predictDuration << " ms" << std::endl;
    checkAccuracy(testingData);


    return EXIT_SUCCESS;
}