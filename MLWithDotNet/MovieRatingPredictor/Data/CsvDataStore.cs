using System;
using System.IO;
using MovieRatingPredictor.Enums;
using MovieRatingPredictor.Interface;
using MovieRatingPredictor.Model;
using Microsoft.ML;
using MovieRatingPredictor.ML;

namespace MovieRatingPredictor.Data
{
    public class CsvDataStore:IDataStore
    {
        private readonly string _trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
        private readonly string _testingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

        public ProcessedData GetTestingData(MLContext mlContext)
        {
            return new ProcessedData()
            {
                Value = mlContext.Data.LoadFromTextFile<MovieRating>(_testingDataPath, hasHeader: true, separatorChar: ',')
            };
        }

        public ProcessedData GetTrainingData(MLContext mlContext)
        {
            return new ProcessedData()
            {
                Value = mlContext.Data.LoadFromTextFile<MovieRating>(_trainingDataPath, hasHeader: true, separatorChar: ',')
            };
        }
    }
}
