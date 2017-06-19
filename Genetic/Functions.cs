using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.Genetic
{
    public delegate double FitnessFunction<TGenome>(Organism<TGenome> org);
    public delegate Organism<TGenome> SpawnFunction<TGenome>(int index);

    public static class CommonGeneticFunctions
    {
        public static SpawnFunction<TGenome> CreateSpawningCycleArray<TGenome>(Organism<TGenome>[] array)
        {
            return new SpawnFunction<TGenome>(index => array[index % array.Length] );
        }
    }

}
