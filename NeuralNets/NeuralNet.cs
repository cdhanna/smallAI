using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.NeuralNets
{

    public class NeuralNet
    {

        private NeuralNetConfig _config;
        private double[] _weights;
        private double[] _biases;

        public int WeightCount
        {
            get
            {
                return 
                 // input layers to first hidden layer column
                  _config.InputLabels.Length * _config.HiddenLayerNeuronCount

                // last hidden layer column to output layer
                + _config.OutputLabels.Length * _config.HiddenLayerNeuronCount

                // hidden layers to hidden layers
                + (_config.HiddenLayerCount > 1
                    ? (_config.HiddenLayerCount - 1) * _config.HiddenLayerNeuronCount * _config.HiddenLayerNeuronCount
                    : 0
                   );
            }
        }

        public int BiasCount
        {
            get
            {
                return _config.HiddenLayerCount * _config.HiddenLayerNeuronCount;
            }
        }

        public NeuralNet(NeuralNetConfig config)
        {
            _config = new NeuralNetConfig(config);
        }

        public void InitializeWeightsToZero()
        {
            InitializeWeights(new double[WeightCount]);
        }

        public void InitializeWeights(double[] weights)
        {
            var expectedCount = WeightCount;
            if (weights.Length != expectedCount)
            {
                throw new InvalidOperationException($"Weight Count Incorrect. Expected {expectedCount} but got {weights.Length}");
            }

            _weights = weights;
        }

        public void InitializeBiases(double[] biases)
        {
            var expectedCount = BiasCount;
            if (biases.Length != expectedCount)
            {
                throw new InvalidOperationException($"Bias Count Incorrect. Expected {expectedCount} but got {biases.Length}");
            }
            _biases = biases;
        }

        public NamedValue[] Compute(NamedValue[] Inputs)
        {

            var nodeCount = _config.HiddenLayerCount * _config.HiddenLayerNeuronCount;
            var nodeSums = new double[nodeCount].Select(d => 0d).ToArray(); // initialize all to 0

            // pass each input value to every node in the first layer
            Inputs.ToList().ForEach(input =>
            {
                for (int i = 0; i < _config.HiddenLayerNeuronCount; i++)
                {
                    nodeSums[i] += input.Value * _weights[i];
                }
            });

            // the soul. Send signal through hidden layers
            for (int columnIndex = 0; columnIndex < _config.HiddenLayerCount - 1; columnIndex++)
            {
                for (int columnNeuronIndex = 0; columnNeuronIndex < _config.HiddenLayerNeuronCount; columnNeuronIndex++)
                {
                    var currentNodeIndex = (_config.HiddenLayerNeuronCount * columnIndex) + columnNeuronIndex;


                    var contributors = _config.HiddenLayerNeuronCount;
                    if (columnIndex == 0) contributors = Inputs.Length;
                    var sum = nodeSums[currentNodeIndex];
                    var normalizedSum = _config.PreActivationNormalizationFunction(contributors, sum);
                    var signal = _config.ActivationFunction(normalizedSum + _biases[currentNodeIndex]);

                    for (int nextColumnNeuronIndex = 0; nextColumnNeuronIndex < _config.HiddenLayerNeuronCount; nextColumnNeuronIndex++)
                    {
                        var nextNodeIndex = (_config.HiddenLayerNeuronCount * (columnIndex + 1)) + nextColumnNeuronIndex;
                        var weightIndex = 
                            // all weights from input layer to first hidden layer
                            (Inputs.Length * _config.HiddenLayerNeuronCount) 
                            // all weights prior to current column
                            + (_config.HiddenLayerNeuronCount * _config.HiddenLayerNeuronCount * columnIndex)
                            // all weights prior to current neuron
                            + (columnNeuronIndex * _config.HiddenLayerNeuronCount)
                            // and the current neuron
                            + (nextColumnNeuronIndex);

                        nodeSums[nextNodeIndex] += _weights[weightIndex] * signal;
                    }
                }
            }

            // last column to output
            var outputs = new List<NamedValue>();
            for (int nameIndex = 0; nameIndex < _config.OutputLabels.Length; nameIndex += 1)
            {
                var outputName = _config.OutputLabels[nameIndex];
                var outputValue = 0d;

                for (int i = 0; i < _config.HiddenLayerNeuronCount; i++)
                {
                    outputValue += nodeSums[i] * _weights[_weights.Length - (_config.OutputLabels.Length) + nameIndex];
                }

                outputValue = _config.PreActivationNormalizationFunction(_config.HiddenLayerNeuronCount, outputValue);

                outputs.Add(new NamedValue() { Name = outputName, Value = outputValue });
            }
            
            return outputs.ToArray();
        }

    }
}
