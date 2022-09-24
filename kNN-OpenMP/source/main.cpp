#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>

int main()
{
    omp_set_num_threads(32);

    std::cout << "Starting..." << std::endl << "Reading dataset..." << std::endl;
    constexpr char delimiter = ',';
    auto reader = DataReader{ "../data/dataset.csv", LabelIndex::LAST, delimiter };
    auto data = reader.readData(10000);

    std::cout << data.size() << " rows read. " << std::endl << std::endl;

    {
        std::cout << "Performing Min-Max Normalization..." << std::endl;
        const auto duration = runWithTimeMeasurementCpu([&]() {
            MinMaxNormalizer{}.normalize(data);
            });
        std::cout << "Data Normalization Completed: " << duration << " ms" << std::endl << std::endl;
    }
    {
        std::cout << "Splitting Dataset..." << std::endl;

        auto [trainingData, testingData] = splitData(data, 90);

        std::cout << "Training Size : " << trainingData.size() << " rows" << std::endl;
        std::cout << "Testing Size  : " << testingData.size() << " rows" << std::endl << std::endl;

        std::cout << "Performing KNN Prediction [OpenMP]......" << std::endl;
        const auto duration = runWithTimeMeasurementCpu([&]() {
            kNNClassifier classifier{ trainingData };
            classifier.predict(testingData);
            });

        std::cout << "Prediction Completed: " << duration << " ms" << std::endl;
        checkAccuracy(testingData);
    }
    return EXIT_SUCCESS;
}

