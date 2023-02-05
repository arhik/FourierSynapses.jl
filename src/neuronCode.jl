
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Permutations
using Images
using Plots

# Dataset
dataset = []
datasetLength = 100

for i in 1:datasetLength
	p = 2.0*rand(32) .- 1.0
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

opt = Descent(0.01f0)

posMean = zeros(32)
negMean = zeros(32) 
dPosMean = 0.0 #.*ones(outdims) # Active mean
dNegMean = 0.0 #.*ones(outdims) # InActive mean

threshold = 0.0

alignmentList = []

thresholdList = []

for epoch in 1:2000
	global posMean, negMean
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	dendriteAlign = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)), :, 1)
	dendriteActivations = dendriteAlign .> posMean
	dendriteActivationCoeffs = fwht(dendriteActivations |> toSpins, ndims(dendriteActivations))
	(dAlign, dendriteGradsBack) = pullback(dendriteActivationCoeffs, dendriteCoeffs, label) do x, y, l
		alignment = abs(sum((x.*y)))
		(alignment)^2
		# alignment
	end

	# alignment = sum(dendriteActivationCoeffs.*dendriteCoeffs)
	# dAlign = alignment
	
	push!(alignmentList, dAlign)


	global dPosMean, dNegMean
	if dAlign < threshold
		dNegMean = (dNegMean + dAlign)/2
	else
		dPosMean = (dPosMean + dAlign)/2
	end

	push!(thresholdList, threshold)

	threshold = (dPosMean + dNegMean)/2
	
	# dendriteGrads = dendriteGradsBack(1)[1]

	activation = dAlign > threshold
	
	# We dont tree grads yet but it would be interesting to speed up learning.
	# treeAlignCoeffs .+= grads[1].*opt.eta
	# treeAlignCoeffs .= treeAlignCoeffs./(sum(treeAlignCoeffs.^2, dims=ndims(treeAlignCoeffs)).^0.5)
	# treeAlignActivations = ifwht(treeAlignCoeffs) |> toBits
	
	if activation
		(sAlign, back) = pullback(datumCoeffs, synapseCoeffs, dendriteActivations) do x, y, dActivations
			sAlign = view(sum(x.*y, dims=ndims(x)), :, 1)
			(dendriteActivations .- sAlign).^2
		end
		global negMean, posMean
		# for l in dendriteActivations
			# if l == 1
				# negMean = (negMean + alignment)/2
			# else
				# posMean = (posMean + alignment)/2
			# end
		# end
		posMean .= (posMean .+ sAlign)/2
		tgrads = back(1)
		synapseCoeffs .-= tgrads[2].*opt.eta
		synapseCoeffs .= synapseCoeffs./(sum(synapseCoeffs.^2, dims=ndims(datumView)).^0.5)
	end
end

threshold = (dPosMean + dNegMean)/2

matches = 0
for epoch in 1:(datasetLength)
	idx = epoch
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumCoeffs = fwht(datumView |> toSpins, ndims(datumView))
	synapseAlign = view(sum(datumCoeffs.*synapseCoeffs, dims=ndims(datumView)), :, 1)
	dAlignActivations = synapseAlign .> posMean
	dAlignCoeffs = fwht(dAlignActivations |> toSpins, ndims(dAlignActivations))
	alignment = sum(dAlignCoeffs.*dendriteCoeffs, dims=ndims(dAlignCoeffs))[1] |> abs
	# activation = (label - alignment).^2 > threshold
	activation = alignment > threshold
	@info alignment, label, Int(activation)
	if label == Int(activation)
		matches += 1
	end
end

print(matches)

plot(alignmentList, color=:lightblue, label="alignment")

plot!(thresholdList, color=:darkmagenta, label="threshold")


