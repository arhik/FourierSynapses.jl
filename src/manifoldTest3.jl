
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Images


# Dataset
dataset = []
datasetLength = 100

for i in 1:datasetLength
	p = 2.0(rand(32) .- 1.0)
	push!(dataset, (p, BitArray(rand(Bool, 32))))
end

# Utility functions
toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = Int8(-1)
@inline toSpins(a::Val{false}) = Int8(1)
toSpins(a::BitVector) = toSpins.(a)
toSpins(a::BitMatrix) = toSpins.(a)
toSpins(b::BitViewArray) = toSpins.(b)
toSpins(a::AbstractArray) = toSpins.(a)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(a)

using FFTW
using Hadamard

synapses = BitArray(rand(Bool, 32, 8*sizeof(eltype(dataset[1][1]))))
dendrites = BitArray(rand(Bool, 32))

synapseCoeffs = fwht(synapses |> toSpins, ndims(synapses))
dendriteCoeffs = fwht(dendrites |> toSpins, ndims(dendrites))

opt = Descent(0.01f0)

posMean = zeros(32)
negMean = zeros(32)

for epoch in 1:6000
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	(alignment, gradsBack) = pullback(datumCoeffs, synapseCoeffs, label) do x, y, l
		alignment = sum(x.*y, dims=ndims(x))
		(l .- alignment).^2
	end
	alignment = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumCoeffs)), :, 1)
	# negMean .= (negMean .+ alignment)/2
	posMean .= (posMean .+ alignment)/2
	grads = gradsBack(1)
	synapseCoeffs .-= grads[2]*opt.eta
	# dendCoeffs[nextpow(2, length(dendCoeffs)/4) : end] .= 0
	synapseCoeffs .= synapseCoeffs/(norm(synapseCoeffs))
end
# 
for epoch in 1:datasetLength
	idx = epoch
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	synapseAlign = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)), :, 1)
	dendriteActivations = synapseAlign .> posMean
	dendriteActivationCoeffs = fwht(dendriteActivations |> toSpins, ndims(dendriteActivations))
	labelCoeffs = fwht(label |> toSpins)
	@show dot(dendriteActivationCoeffs, labelCoeffs)
end

# threshold = (posMean + negMean)/2
# 
# testresult = []
# 
# for epoch in 1:1000
	# idx = epoch
	# (datum, label) = dataset[idx]
	# datumview = bitview(datum)
	# datumCoeffs = fwht(datumview[:] |> toSpins, 1)
	# push!(testresult, (datum, dot(datumCoeffs, dendCoeffs) > threshold))
# end

# testdataset = []
# 
# for i in 1:10000
	# p = 2.0*rand(2) .- 1.0
	# if norm(p) < 0.5
		# push!(testdataset, (p, 0))
	# else
		# push!(testdataset, (p, 1))
	# end
# end
# 
# using Plots
# 
# scatter(map(x -> first(x) |> first, testdataset), map(x -> first(x) |> last, testdataset), markercolor=map(c -> ifelse(c == 0, :blue, :red), map(last, testdataset)))
# 
# testresult = []
# 
# for epoch in 1:10000
	# idx = epoch
	# (datum, label) = testdataset[idx]
	# datumview = bitview(datum)
	# datumCoeffs = fwht(datumview[:] |> toSpins, 1)
	# push!(testresult, (datum, dot(datumCoeffs, dendCoeffs) > threshold))
# end
# 
# using Plots
# 
# scatter(map(x -> first(x) |> first, testresult), map(x -> first(x) |> last, testresult), markercolor=map(c -> ifelse(c == 0, :blue, :red), map(last, testresult)))


