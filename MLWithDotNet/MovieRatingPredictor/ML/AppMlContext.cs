using Microsoft.ML;

namespace MovieRatingPredictor.ML
{
    public class AppMlContext
    {
        public ITransformer Model { get; set; }
        public MLContext MLContext { get; set; }

        public IEstimator<ITransformer> Trainer { get; set; }

    }
}
