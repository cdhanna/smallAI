using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.Genetic
{
    public class GeneticSimulation<TGenome>
    {

        private FitnessFunction<TGenome> _fitnessFunction;
        private SpawnFunction<TGenome> _spawnFunction;

        private List<Organism<TGenome>> _population;
        private Organism<TGenome> _best;

        private Random _rand;

        public GeneticSimulation(FitnessFunction<TGenome> fitnessFunction, SpawnFunction<TGenome> spawnFunction, int populationSize)
        {
            _population = new List<Organism<TGenome>>();
            _rand = new Random();

            _fitnessFunction = fitnessFunction;

            for (int i = 0; i < populationSize; i++)
            {
                _population.Add(spawnFunction(i));
            }

        }

        public Organism<TGenome> GetBest()
        {
            return _best;
        }

        public List<Organism<TGenome>> Evolve(int generations, bool rampDownMutation=true)
        {
            if (rampDownMutation)
            {
                for (int i = 0; i < generations; i++)
                {
                    Evolve( (generations - i) / (double)generations);
                }
            } else
            {
                for (int i = 0; i < generations; i++)
                {
                    Evolve();
                }
            }


            return _population;
        }
        public List<Organism<TGenome>> Evolve(double mutationMultiplier=1)
        {
            // pass each member through the fitness function
            List<double> scores = _population.Select(org => _fitnessFunction(org)).ToList();

            // take the top 50% ish of the list
            var bestHalf = new List<Organism<TGenome>>();
            var averageScore = scores.Average();
            //var scoreThreshold = (scores.Min() + scores.Max()) ;
            var bestScoreSoFar = double.MinValue;
            for (int i = 0; i < _population.Count; i++)
            {
                
                if (scores[i] >= averageScore && bestHalf.Count < _population.Count/2)
                {
                    bestHalf.Add(_population[i]);
                }
                if (scores[i] > bestScoreSoFar)
                {
                    bestScoreSoFar = scores[i];
                    _best = _population[i];
                }
            }

            // to generate the other half, randomly pick organisms from the bestHalf, and breed them
            var newOrganisms = new List<Organism<TGenome>>();
            if (bestHalf.Count == 0)
                bestHalf = _population.Take(_population.Count / 2).ToList();
            Trace.WriteLine("Generating new population "+ bestScoreSoFar);
            while (newOrganisms.Count  + bestHalf.Count < _population.Count)
            {
                var momIndex = _rand.Next(bestHalf.Count);
                var dadIndex = _rand.Next(bestHalf.Count);
                var mom = bestHalf[momIndex];
                var dad = bestHalf[dadIndex];
                //Trace.WriteLine($"{scores[momIndex]},    {scores[dadIndex]}");
                var child = mom.BreedWith(dad, mutationMultiplier);
                newOrganisms.Add(child);
            }

            _population = new List<Organism<TGenome>>();
            _population.AddRange(newOrganisms);
            _population.AddRange(bestHalf);

            return _population;
        }

    }
}
