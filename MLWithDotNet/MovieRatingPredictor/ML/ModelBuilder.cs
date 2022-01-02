using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRatingPredictor.Model;
using System.Linq;

namespace MovieRatingPredictor.ML
{
    public static class ModelBuilder
    {
        public static AppMlContext CreateContext()
        {
            return new AppMlContext()
            {
                MLContext = new MLContext()
            };
        }

        public static AppMlContext MapValueToKey(this AppMlContext context, string inputColumn, string outputColumn)
        {
             context.MLContext.Transforms.Conversion.MapValueToKey(outputColumn, inputColumn);
            return context;
        }

        public static AppMlContext CreateTrainer(this AppMlContext context, MatrixFactorizationTrainer.Options options)
        {
            context.Trainer = context.MLContext.Recommendation().Trainers.MatrixFactorization(options);
            return context;
        }

        public static AppMlContext Train(this AppMlContext context, IDataView trainingDataView)
        {
            context.Model = context.Trainer.Fit(trainingDataView);
            return context;
        }

        public static MovieRatingPrediction UseModelForSinglePrediction(this AppMlContext context, MovieRating testInput)
        {
            var predictionEngine = context.MLContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(context.Model);
            return predictionEngine.Predict(testInput);
        }
    }
}
