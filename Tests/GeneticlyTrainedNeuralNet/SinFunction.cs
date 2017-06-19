using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Small.Ai.Genetic;
using Small.Ai.NeuralNets;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace Tests.GeneticlyTrainedNeuralNet
{
    [TestClass]
    public class SinFunction
    {

        class Org : Organism<double[]>
        {
            Random rand = new Random();

            public Org(double[] genome)
            {
                Genome = genome;
            }

            public override Organism<double[]> BreedWith(Organism<double[]> mate, double mutationMultiplier)
            {
                var maxIndex = Math.Max(Genome.Length, mate.Genome.Length);
                var outputGenome = new double[maxIndex];
                for (int i = 0; i < maxIndex; i++)
                {
                    var selfContrib = i < Genome.Length ? Genome[i] : 0;
                    var mateContrib = i < mate.Genome.Length ? mate.Genome[i] : 0;
                    var mutation = (rand.NextDouble()-.5) * mutationMultiplier * .2;
                    var pref = 1;
                    outputGenome[i] = ((1 - pref) * selfContrib) + (pref * mateContrib);
                    //outputGenome[i] = Math.Max(Math.Min(outputGenome[i], 1), 0);
                }
                for (int i = 0; i < maxIndex / 2; i++)
                {
                    var randomIndex = rand.Next(maxIndex);
                    outputGenome[randomIndex] += (rand.NextDouble() - .5);
                    outputGenome[i] = Math.Max(Math.Min(outputGenome[i], 1), 0);
                }
                return new Org(outputGenome);
            }

            public override Organism<double[]> Clone()
            {
                return new Org(Genome);
            }
        }


        [TestMethod]
        public void Sin()
        {
            var rand = new Random();
            var net = NeuralNetBuilder.From(new NeuralNetConfig() {
                InputLabels = new string[] { "x" },
                OutputLabels = new string[] { "y" },
                HiddenLayerCount = 3,
                HiddenLayerNeuronCount = 3,
                //PreActivationFunction = (i, d) => 
                ActivationFunction = d => d > .5 ? 1 : 0
                //ActivationFunction = d => 1 / (1 + Math.Pow(Math.E, -d))
            });
            var trainingData = new List<Tuple<double, double>>();
            for (double i = 0; i < Math.PI * 2; i += .1)
            {
                var input = i;
                var output = Math.Sin(i);
                //var normalizedOutput = output / Math.PI * 2;
                var normalizedOutput = .5 * (output + 1);
                var normalizedInput = input / (Math.PI * 2);
                trainingData.Add(new Tuple<double, double>(normalizedInput, normalizedOutput));
            }

            var weightCount = net.WeightCount;
            var biasCount = net.BiasCount;
            var weightOrgSpawner = new SpawnFunction<double[]>(index =>
            {
                var data = new double[weightCount + biasCount];
                for (int i = 0; i < weightCount + biasCount; i++)
                {
                    var scale = i >= weightCount ? 1 : 1;
                    data[i] = (rand.NextDouble()) * scale;
                }

                return new Org(data);
            });

            var fitnessFunction = new FitnessFunction<double[]>(org =>
            {
                net.InitializeWeights(org.Genome.Take(weightCount).ToArray() );

                net.InitializeBiases(org.Genome.Skip(weightCount).ToArray());

                var errorSum = trainingData.Select(trainingElement =>
                {
                    var input = trainingElement.Item1;
                    var expectedOutput = trainingElement.Item2;

                    var actualOutput = net.Compute(new NamedValue[] { new NamedValue() { Name = "x", Value = input } });

                    var error = expectedOutput - actualOutput[0].Value;
                    var convertedError = Math.Pow((Math.Abs(error) + 1), 2);

                    return convertedError;
                }).Sum();
                //Trace.WriteLine(errorSum);
                return 1/errorSum;
            });

            var geneticSim = new GeneticSimulation<double[]>(fitnessFunction, weightOrgSpawner, 50);

            var bestSet = geneticSim.Evolve(20000, false);
            var theBest = geneticSim.GetBest();

            net.InitializeWeights(theBest.Genome.Take(weightCount).ToArray());
            net.InitializeBiases(theBest.Genome.Skip(weightCount).ToArray());

            Trace.WriteLine("OUTPUT");
            trainingData.ForEach(t =>
            {
                var input = t.Item1;
                var expectedOutput = t.Item2;

                var actualOutput = net.Compute(new NamedValue[] { new NamedValue() { Name = "x", Value = input } });

                var error = expectedOutput - actualOutput[0].Value;

                //Trace.WriteLine($"INPUT:{input} EXPECT:{expectedOutput} REAL:{actualOutput[0].Value} ERR:{error}");
                //Trace.WriteLine($"{input},{expectedOutput},{actualOutput[0].Value}");
                Trace.WriteLine($"{actualOutput[0].Value}");
            });
        }
    }
}
