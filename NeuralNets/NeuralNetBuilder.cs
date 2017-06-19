using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Small.Ai.NeuralNets
{
    public static class NeuralNetBuilder
    {
        public static NeuralNet From(NeuralNetConfig config)
        {
            return new NeuralNet(config);
        }
    }
}
