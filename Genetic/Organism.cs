using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.Genetic
{
    public abstract class Organism<TGenome>
        //where TGenome : ICloneable
    {
        public TGenome Genome { get; protected set; }

        public abstract Organism<TGenome> BreedWith(Organism<TGenome> mate, double mutationMultiplier);
        public abstract Organism<TGenome> Clone();
    }
}
