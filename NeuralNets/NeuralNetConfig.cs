using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.NeuralNets
{
    public class NeuralNetConfig
    {
        public int HiddenLayerCount { get; set; } = 3;
        public int HiddenLayerNeuronCount { get; set; } = 12;
        public string[] InputLabels { get; set; }
        public string[] OutputLabels { get; set; }

        public Func<int, double, double> PreActivationNormalizationFunction { get; set; } = (i, d) => d / i;
        public Func<double, double> ActivationFunction { get; set; } = d => 1 / (1 + Math.Pow(Math.E, -d));
        public NeuralNetConfig()
        {
            // empty
        }

        public NeuralNetConfig(NeuralNetConfig copy)
        {
            HiddenLayerCount = copy.HiddenLayerCount;
            HiddenLayerNeuronCount = copy.HiddenLayerNeuronCount;
            InputLabels = copy.InputLabels.ToList().ToArray();
            OutputLabels = copy.OutputLabels.ToList().ToArray();
            ActivationFunction = copy.ActivationFunction;
            PreActivationNormalizationFunction = copy.PreActivationNormalizationFunction;
        }

    }
}
