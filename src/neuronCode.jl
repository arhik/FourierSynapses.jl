
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Permutations
using Images

# Dataset
dataset = []

for i in 1:100
	p = (rand(N0f8, 32, ) .- 0.5N0f8)
	if rand() < 0.5
		push!(dataset, (p, 0))
	else
		push!(dataset, (p, 1))
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

toBits(a::Number) = a < 0
toBits(a::AbstractArray) = toBits.(a)

using FFTW
using Hadamard

outdims = 32

synapses = BitArray(rand(Bool, 32, 8*sizeof(eltype(dataset[1][1]))))
dendrites = BitArray(rand(Bool, 32,))

synapseCoeffs = fwht(synapses |> toSpins, ndims(synapses))
dendriteCoeffs = fwht(dendrites |> toSpins, ndims(dendrites))

opt = Descent(0.001f0)

posMean = 0.0
negMean = 0.0 
dPosMean = 0.0 #.*ones(outdims)
dNegMean = 0.0 #.*ones(outdims)

for epoch in 1:50000
	global posMean, negMean
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	synapseAlign = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)), :, 1)
	treeAlignActivations = synapseAlign .> posMean
	treeAlignCoeffs = fwht(treeAlignActivations |> toSpins, ndims(treeAlignActivations))
	treegrads = gradient(treeAlignCoeffs, dendriteCoeffs, label) do x, y, l
		alignment = sum((x.*y))
		(l - alignment)^2
	end
	global dPosMean, dNegMean
	alignment = sum(treeAlignCoeffs.*dendriteCoeffs, dims=ndims(treeAlignCoeffs))[1]
	if label == 0
		dNegMean = (dNegMean + alignment)/2
	else
		dPosMean = (dPosMean + alignment)/2
	end
	threshold = (dPosMean + dNegMean)/2
	activation = alignment > threshold
	# We dont tree grads yet but it would be interesting to speed up learning.
	# treeAlignCoeffs .+= grads[1].*opt.eta
	# treeAlignCoeffs .= treeAlignCoeffs./(sum(treeAlignCoeffs.^2, dims=ndims(treeAlignCoeffs)).^0.5)
	# treeAlignActivations = ifwht(treeAlignCoeffs) |> toBits
	if activation
		grads = pullback(datumCoeffs, synapseCoeffs, activation) do x, y, l
			synapseAlign = view(sum(x.*y, dims=ndims(x)), :, 1)
			global negMean, posMean
			if l == 0
				negMean = (negMean + alignment)/2
			else
				posMean = (posMean + alignment)/2
			end
			(Int(l) .- synapseAlign).^2
		end
		synapseCoeffs .-= grads[1].*opt.eta
		synapseCoeffs .= synapseCoeffs./(sum(synapseCoeffs.^2, dims=ndims(datumView)).^0.5)
	end
end

matches = 0
for epoch in 1:100
	idx = epoch
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	synapseAlign = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)), :, 1)
	treeAlignActivations = synapseAlign .> posMean
	treeAlignCoeffs = fwht(treeAlignActivations |> toSpins, ndims(treeAlignActivations))
	alignment = sum(treeAlignCoeffs.*dendriteCoeffs, dims=ndims(treeAlignCoeffs))[1]
	activation = alignment > threshold
	@info label, Int(activation)
	if label == Int(activation)
		matches += 1
	end
end

print(matches)

threshold = (dPosMean + dNegMean)/2

testresult = []

for epoch in 1:100
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	push!(testresult, (datum, sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)) .> threshold))
end


using Plots

scatter(map(first, testresult), map(last, testresult))

# Cartesian Indices 
using Permutations
synapses = BitArray(undef, (32, 64))
idxs = CartesianIndices((1:32, 1:64))
perm = RandomPermutation(32*64)
permTransform = two_row(perm)
permIdxs = idxs[permTransform[2, :]]
out = synapses[permIdxs]
reshape(out, synapses |> size)

