using System;
using System.Collections.Generic;
using System.IO;
using SOPAIDwrapper;

namespace SOPAID_wrapper_test
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            Console.WriteLine("SOPAID CLR wrapper smoke test");
            Console.WriteLine("Process: " + (Environment.Is64BitProcess ? "x64" : "x86"));

            if (args.Length == 0)
            {
                Console.WriteLine();
                Console.WriteLine("Usage:");
                Console.WriteLine("  SOPAID_wrapper_test.exe <model-path>");
                Console.WriteLine();
                Console.WriteLine("Wrapper load test passed. Provide a model path to test native Init().");
                return 0;
            }

            string modelPath = args[0];
            if (!File.Exists(modelPath))
            {
                Console.WriteLine("Model file not found: " + modelPath);
                return 2;
            }

            var config = new InferenceConfig
            {
                ModelPath = modelPath,
                Format = ModelFormat.Auto,
                InputWidth = 640,
                InputHeight = 640,
                ConfidenceThreshold = 0.25f,
                NmsThreshold = 0.70f,
                ClassNamesCsv = "bearing,cover,tool",
                UseCuda = false,
                DeviceId = 0
            };

            using (var evaluator = new InferenceEvaluator(config))
            {
                Console.WriteLine("Init result: " + evaluator.IsInitialized);
                Console.WriteLine("Status: " + evaluator.LastError.Status);

                if (!string.IsNullOrWhiteSpace(evaluator.LastError.Message))
                {
                    Console.WriteLine("Message: " + evaluator.LastError.Message);
                }

                if (!evaluator.IsInitialized)
                {
                    return 3;
                }

                var results = new List<DetectionResult>();
                byte[] dummyImage = new byte[640 * 480 * 3];

                bool ok = evaluator.Evaluate(dummyImage, 640, 480, 3, 640 * 3, results);
                Console.WriteLine("Evaluate dummy image: " + ok);
                Console.WriteLine("Status: " + evaluator.LastError.Status);

                if (!string.IsNullOrWhiteSpace(evaluator.LastError.Message))
                {
                    Console.WriteLine("Message: " + evaluator.LastError.Message);
                }

                Console.WriteLine("Detection count: " + results.Count);
                return ok ? 0 : 4;
            }
        }
    }
}
