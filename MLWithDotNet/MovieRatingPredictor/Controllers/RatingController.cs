using MovieRatingPredictor.Data;
using MovieRatingPredictor.Enums;
using MovieRatingPredictor.Model;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using MovieRatingPredictor.ML;
using Microsoft.ML.Trainers;
using MovieRatingPredictor.Interface;

namespace MovieRatingPredictor.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class RatingController : ControllerBase
    {
        private IDataStore _dataStore;
        private AppMlContext _context;

        public RatingController(IDataStore dataStore)
        {
            _dataStore = dataStore;
        }

        [HttpGet]
        public IActionResult PredictRating()
        {
            var metric = _context.Model
                        .EvaluateModel(_dataStore.GetTestingData(_context.MLContext).Value);
            return Ok(metric);
        }

        [HttpPost]
        public IActionResult BuildModel()
        {
            _context = ModelBuilder.CreateContext()
                .MapValueToKey("userId", "userIdEncoded")
                .MapValueToKey("movieId", "movieIdEncoded");

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            _context.CreateTrainer(options)
               .Train(_dataStore.GetTrainingData(_context.MLContext).Value);
            return Ok();
        }
    }
}