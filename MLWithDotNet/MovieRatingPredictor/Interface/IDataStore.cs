using Microsoft.ML;
using MovieRatingPredictor.ML;

namespace MovieRatingPredictor.Interface
{
    public interface IDataStore
    {
        ProcessedData GetTrainingData(MLContext mlContext);
        ProcessedData GetTestingData(MLContext mlContext);
    }
}