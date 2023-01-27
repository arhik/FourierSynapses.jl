
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Permutations

# Dataset
dataset = []

for i in 1:1000
	p = 2.0*rand() - 1.0
	if norm(p) < 0.5
		push!(dataset, (p, false))
	else
		push!(dataset, (p, true))
	end
end

# Utility functions
toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1 |> Int8
@inline toSpins(a::Val{false}) = +1 |> Int8
toSpins(a::BitVector) = toSpins.(a)
toSpins(a::BitMatrix) = toSpins.(a)
toSpins(b::BitViewArray) = toSpins.(b)
toSpins(b::AbstractArray) = toSpins.(b)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

using FFTW
using Hadamard

dendrite = BitArray(undef, (32, 64))

dendCoeffs = fwht(dendrite |> toSpins)

opt = Descent(0.001f0)

posMean = 0
negMean = 0

for epoch in 1:20000
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	grads = gradient(datumCoeffs, dendCoeffs, label) do x, y, l
		alignment = dot(x, y)
		global posMean, negMean
		if l == 0
			negMean = (negMean + alignment)/2
		else
			posMean = (posMean + alignment)/2
		end 
		(l-alignment)^2
	end
	dendCoeffs .-= grads[2]*opt.eta
	dendCoeffs .= dendCoeffs/(norm(dendCoeffs))
end

for epoch in 1:1000
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	@show dot(datumCoeffs, dendCoeffs), label
end

testresult = []

for epoch in 1:1000
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	@show push!(testresult, (datum, dot(datumCoeffs, dendCoeffs) > threshold))
end

threshold = (posMean + negMean)/2

using Plots

scatter(map(first, testresult), map(last, testresult))

# Cartesian Indices 
using Permutations

idxs = CartesianIndices((1:32))

perm = RandomPermutation(32)

permTransform = two_row(p)

permIdxs = idxs[pt[2, :]]
