
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Permutations
using Images
using Plots
using FFTW
using Hadamard


# Utility functions
toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1 |> Int8
@inline toSpins(a::Val{false}) = +1 |> Int8
toSpins(a::BitVector) = toSpins.(a)
toSpins(a::BitMatrix) = toSpins.(a)
toSpins(b::BitViewArray) = toSpins.(b)
toSpins(b::AbstractArray) = toSpins.(b)

toSpins(a::Number) = toSpins(a .> 0)

toBits(a::Number) = a < 0
toBits(a::AbstractArray) = toBits.(a)


# XOR function 

nSynapses = 8
nDendrites = 2

dOut(a, x) = (prod((1 .- (a).*(x))/2, dims=1))

a = randn(Float32, nSynapses, nDendrites)
x = rand(Bool, nSynapses, nDendrites) .|> toSpins
y = rand(Bool, nDendrites) |> toSpins

b = rand(Float32, 1, nDendrites)
dOut(b, y)

y, back = pullback(dOut, a, x)
back(1)

dataset = []
# Radius dataset
for i in 1:10 # TODO arbitrary for now
	p = rand(2) .|> N0f8
	if 0.2 < norm(p) < 0.7
		push!(dataset, (p, 0))
	else
		push!(dataset, (p, 1))
	end
end

opt = Descent(1e-5)

fPosMean = 0.0
fNegMean = 0.0

# This loop repeats 'repeats' times for each data point on average
repeats = 1000
for epoch in 1:(length(dataset)*repeats)
	global fPosMean, fNegMean
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumSpins = datumView |> toSpins |> adjoint |> collect
	dActs = dOut(a, datumSpins) .|> toSpins
	nActs = dOut(b, dActs) .|> toSpins
	
	(nOuts, nBack) = pullback(b, dActs, label |> toSpins) do bb, da, l
		(l .- dOut(bb, da)).^2
	end	
	
	(dOuts, dBack) = pullback(a, datumSpins, dActs) do aa, da, l
		(l .- dOut(aa, da)).^2
	end

	plot!(nOuts)

	(grads, _) = nBack(1)
	(grads2, _) = dBack(1)
	b .+= toSpins(label).*grads*opt.eta
	a .+= toSpins(nActs).*toSpins(dActs).*grads2*opt.eta
end

# threshold = (fPosMean + fNegMean)/2

matches = 0
for epoch in 1:(dataset |> length)
	idx = epoch
	(datum, label) = dataset[idx]
	datumView = bitview(datum)
	datumSpins = datumView |> toSpins |> adjoint |> collect
	dActs = dOut(a, datumSpins) .|> toSpins
	nActs = dOut(b, dActs) .|> toSpins
	@info first(nActs), dActs, datumSpins, label
	
	if toSpins(label) == Int(nActs[1])
		matches += 1
	end
end

print(matches/length(dataset))

# plot(alignmentList, color=:lightblue, label="alignment")
# 
# plot!(thresholdList, color=:darkmagenta, label="threshold")
# 
