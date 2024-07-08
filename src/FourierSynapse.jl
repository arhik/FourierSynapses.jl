module FourierSynapse

# This research work explores the idea that synapses can represent boolean
# fourier transform. The idea is that each dendrite is computing an equivalent
# of boolean function. And the neuron is computing another fourier transform or
# just a weighted sum of dendritic inputs. This approach has potential to be both
# memory-efficient & compute-efficient on CPU & GPU hardware. Not only does this
# approach encode an associative but also has potential to decompose input into
# causal and confounding associations because of its structure and metrics.
#
# Early experiments proved gradient vanishing is a problem and need innovation
# on gradient vanishing.
#
# This idea was from 2017 experiments. Reconsidering this after gaussian splatting
# paper. We had experiments without Gaussian assumptions. But 3DGS has shown how
# cooperative groups can speed up computations. We can redo these experiments and
# speed up training. Gaussian splatting itself would be an interesting application
# for this approach. Hopefully this would be more efficient for its contextual
# splatting capability.

# N.B Current code doesn't reflect the fourier transform but instead use hadamard'

end # module FourierSynapse
