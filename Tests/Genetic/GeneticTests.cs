using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Small.Ai.Genetic;
using System.Linq;
using System.Collections.Generic;

namespace Tests.Genetic
{
    [TestClass]
    public class GeneticTests
    {
        
        class SampleOrg : Organism<double[]>
        {
            Random rand = new Random();
            public SampleOrg(double[] genome)
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
                    var mutation = (rand.NextDouble() - .5) * mutationMultiplier;
                    outputGenome[i] = mutation + ((selfContrib + mateContrib) / 2);
                }
                return new SampleOrg(outputGenome);
            }

            public override Organism<double[]> Clone()
            {
                return new SampleOrg(CloneDoubleArray(Genome));
            }

            static double[] CloneDoubleArray(double[] array)
            {
                double[] result = new double[array.Length];
                Buffer.BlockCopy(array, 0, result, 0, array.Length * sizeof(double));
                return result;
            }
        }

        [TestMethod]
        public void Simple()
        {

            var sim = new GeneticSimulation<double[]>(
                org => 1 / (Math.Abs(10 - org.Genome.Sum()) + .01d),
                CommonGeneticFunctions.CreateSpawningCycleArray(new SampleOrg[] {
                    new SampleOrg(new double[] { 1, 2, 3, 0}),
                    new SampleOrg(new double[] { 0, 1, 2}),
                    new SampleOrg(new double[] { 1, 0, 1}),
                }),
                100);

            //var evolved = new List<SampleOrg>();
            //for (int simCount = 0; simCount < 1000; simCount ++)
            //{
            //    evolved = sim.Evolve().Cast<SampleOrg>().ToList();

            //}

            var evolved = sim.Evolve(15);
            var finalSums = evolved.Select(o => o.Genome.Sum()).ToArray();
        }
    }
}
