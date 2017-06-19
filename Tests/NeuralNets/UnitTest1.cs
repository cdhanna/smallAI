using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Small.Ai.NeuralNets;

namespace Tests.NeuralNets
{
    [TestClass]
    public class NeuralNetTests
    {
        [TestMethod]
        public void Compute()
        {
            var net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 1,
                HiddenLayerNeuronCount = 1,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1", "2","3" }
            });
            net.InitializeWeights(new double[1 + 3] { .5, .5, .25, 1 });

            var output = net.Compute(new NamedValue[]
            {
                new NamedValue() {Name="A", Value=1 }
            });


        }

        [TestMethod]
        public void Compute_2Layers()
        {
            var net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 2,
                HiddenLayerNeuronCount = 2,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2 + 2 + 4] {
                1, 1, // input weights
                .5, 1, // neruon 1 to 3 and 4
                1, 1, // neruon 2 to 3 and 4
                1, 1  // output weights
            });

            var output = net.Compute(new NamedValue[]
            {
                new NamedValue() {Name="A", Value=1 }
            });


        }

        [TestMethod]
        public void Compute_3Layers()
        {
            var net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 3,
                HiddenLayerNeuronCount = 2,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2 + 2 + 4 + 4] {
                1, 1, // input weights
                .5, 1, // neruon 1 to 3 and 4
                1, 1, // neruon 2 to 3 and 4
                1, 1, 
                1, 1,
                1, 1  // output weights
            });

            var output = net.Compute(new NamedValue[]
            {
                new NamedValue() {Name="A", Value=1 }
            });
        }

        [TestMethod]
        public void CorrectWeightCount()
        {
            var net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 1,
                HiddenLayerNeuronCount = 1,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2]);

            net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 1,
                HiddenLayerNeuronCount = 2,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2 + 2]);

            net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 2,
                HiddenLayerNeuronCount = 2,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2 + 2 + 4]);

            net = NeuralNetBuilder.From(new NeuralNetConfig()
            {
                HiddenLayerCount = 3,
                HiddenLayerNeuronCount = 2,
                InputLabels = new string[] { "A" },
                OutputLabels = new string[] { "1" }
            });
            net.InitializeWeights(new double[2 + 2 + 4 + 4]);
        }
    }
}
