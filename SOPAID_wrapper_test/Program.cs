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
                Console.WriteLine("  SOPAID_wrapper_test.exe <model-path-or-project-directory>");
                Console.WriteLine();
                Console.WriteLine("Wrapper load test passed. Provide a model path to test native Init().");
                return 0;
            }

            string inputPath = args[0];
            bool projectMode = Directory.Exists(inputPath);
            if (!projectMode && !File.Exists(inputPath))
            {
                Console.WriteLine("Model or project directory not found: " + inputPath);
                return 2;
            }

            InferenceEvaluator evaluator = projectMode
                ? new InferenceEvaluator(new ProjectInferenceConfig
                  {
                      ProjectDirectory = inputPath,
                      PreferredFormat = ModelFormat.Auto,
                      UseCuda = false
                  })
                : new InferenceEvaluator(new InferenceConfig
                  {
                      ModelPath = inputPath,
                      Format = ModelFormat.Auto,
                      ClassNamesCsv = "bearing,cover,tool",
                      UseCuda = false
                  });

            using (evaluator)
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

                ModelInfo modelInfo = evaluator.GetModelInfo();
                if (modelInfo == null)
                {
                    Console.WriteLine("GetModelInfo failed: " + evaluator.LastError.Message);
                    return 4;
                }
                Console.WriteLine("Project: " + modelInfo.ProjectId);
                Console.WriteLine("Model: " + modelInfo.ModelId + " " + modelInfo.ModelVersion);
                Console.WriteLine("Backend: " + modelInfo.Backend);
                Console.WriteLine("Classes: " + modelInfo.ClassCount);

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
                var channelResults = new List<DetectionResult>();
                bool grayOk = evaluator.Evaluate(new byte[320 * 240], 320, 240, 1, channelResults);
                bool bgraOk = evaluator.Evaluate(new byte[320 * 240 * 4], 320, 240, 4, channelResults);
                Console.WriteLine("Gray/BGRA conversion: " + grayOk + "/" + bgraOk);
                return ok && grayOk && bgraOk ? 0 : 5;
            }
        }
    }
}
