using Microsoft.ML;
using MovieRatingPredictor.Model;

namespace MovieRatingPredictor.ML
{
    public class ProcessedData
    {
        public IDataView Value { get; set; }
    }

    public class RegressionMetrics : EvaluationMetrics
    {
        public RegressionMetrics(Microsoft.ML.Data.RegressionMetrics metrics)
        {
            RootMeanSquaredError = metrics.RootMeanSquaredError;
        }

        public double RootMeanSquaredError { get; private set; }
    }
}
