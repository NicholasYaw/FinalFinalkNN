#include "DataReader.hpp"
#include "MinMaxNormalizer.hpp"
#include "kNNClassifier.hpp"
#include "FlatDataView.hpp"
#include "HelperFunctions.hpp"
#include <iostream>
#include <chrono>
#include <thread>

struct MinMax
{
    float min;
    float max;

    bool operator==(const MinMax& other)
    {
        constexpr auto epsilon = 10e-4f;
        return (std::abs(min - other.min) < epsilon) && (std::abs(max - other.max) < epsilon);
    }
};

std::vector<MinMax> findFeatureMinMax(const std::vector<DataRow>& dataRows)
{
    const auto numOfFeatures = dataRows.front().features.size();
    std::vector<MinMax> featuresMinMax(numOfFeatures, { std::numeric_limits<float>::max(), std::numeric_limits<float>::min() });

    for (const auto& row : dataRows)
    {
        for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
        {
            const auto feature = row.features[featureIndex];
            auto& featureMinMax = featuresMinMax[featureIndex];

            if (feature < featureMinMax.min)
            {
                featureMinMax.min = feature;
            }

            if (feature > featureMinMax.max)
            {
                featureMinMax.max = feature;
            }
        }
    }

    return featuresMinMax;
}

void singleNormalize(DataRow& row, unsigned numOfFeatures, std::vector<MinMax> minsMaxs) {
    for (auto featureIndex = 0u; featureIndex < numOfFeatures; ++featureIndex)
    {
        row.features[featureIndex] = (row.features[featureIndex] - minsMaxs[featureIndex].min) / (minsMaxs[featureIndex].max - minsMaxs[featureIndex].min);
    }
}


void normalize(std::vector<DataRow>& dataRows)
{
    std::vector<std::thread> threads;
    const auto numOfFeatures = dataRows.front().features.size();
    const auto minsMaxs = findFeatureMinMax(dataRows);

    for (auto& row : dataRows)
    {
        threads.push_back(std::thread(singleNormalize, std::ref(row), std::ref(numOfFeatures), std::ref(minsMaxs)));
    }

    for (auto& th : threads) {
        th.join();
    }
}

double calculateDistance(const DataRow& lhs, const DataRow& rhs)
{
    const auto numOfFeatures = lhs.features.size();
    double sum = 0;

    for (auto i = 0u; i < numOfFeatures; ++i)
    {
        sum += std::pow(lhs.features[i] - rhs.features[i], 2);
    }

    return std::sqrt(sum);
}

void singlePredict(DataRow& testRow, std::vector<DataRow>& trainingData) {
    auto smallestDistance = std::numeric_limits<double>::max();
    auto nearestLabel = 0u;

    for (const auto& trainRow : trainingData)
    {
        const auto distance = calculateDistance(trainRow, testRow);
        if (distance < smallestDistance)
        {
            smallestDistance = distance;
            nearestLabel = trainRow.label;
        }
    }

    testRow.predictedLabel = nearestLabel;
}

void predict(std::vector<DataRow>& trainingData, std::vector<DataRow>& testingData)
{
    std::vector<std::thread> threads;

    for (auto& testRow : testingData)
    {
        threads.push_back(std::thread(singlePredict, std::ref(testRow), std::ref(trainingData)));
    }

    for (auto& th : threads) {
        th.join();
    }
}

int main()
{
        
    //Read Dataset
    std::cout << "Starting..." << std::endl << "Reading dataset..." << std::endl;
    constexpr char delimiter = ',';
    auto reader = DataReader{ "../data/dataset.csv", LabelIndex::LAST, delimiter };
    auto data = reader.readData(10000);
    std::cout << data.size() << " rows read. " << std::endl << std::endl;

    {
        //Normalization
        std::cout << "Performing Min-Max Normalization..." << std::endl;
        const auto duration = runWithTimeMeasurementCpu([&]() {
            MinMaxNormalizer{}.normalize(data);
            });
        std::cout << "Data Normalization Completed: " << duration << " ms" << std::endl << std::endl;
    }

    {
        //Split Dataset
        std::cout << "Splitting Dataset..." << std::endl;
        auto [trainingData, testingData] = splitData(data, 90);
        std::cout << "Training Size : " << trainingData.size() << " rows" << std::endl;
        std::cout << "Testing Size  : " << testingData.size() << " rows" << std::endl << std::endl;

        //Parallel
        std::cout << "Performing KNN Prediction [Parallel Threading]......" << std::endl;
        const auto duration = runWithTimeMeasurementCpu([&]() {
            predict(trainingData, testingData);
        });
        std::cout << "Prediction Completed: " << duration << " ms" << std::endl;

        //Shows Accuracy
        checkAccuracy(testingData);
        std::cout << std::endl << std::endl << std::endl;
    }
    return EXIT_SUCCESS;
}


////Sequential KNN Prediction
    //std::cout << "Performing KNN Prediction [Sequential]......" << std::endl;
    //const auto duration = runWithTimeMeasurementCpu([&]() {
    //    kNNClassifier{ trainingData }.predict(testingData);
    //});