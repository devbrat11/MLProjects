using Microsoft.ML;
using MovieRatingPredictor.Model;

namespace MovieRatingPredictor.ML
{
    public static class ModelEvaluator
    {
        public static IDataView EvaluateModel(this ITransformer model, IDataView testDataView)
        {
            return model.Transform(testDataView);
        }

        public static EvaluationMetrics GetMetrics(this AppMlContext context, IDataView prediction, string labelColumnName, string scoreColumnName)
        {
            var metrics = context.MLContext.Regression.Evaluate(prediction, labelColumnName: labelColumnName, scoreColumnName: scoreColumnName);
            return new RegressionMetrics(metrics);
        }
    }
}
