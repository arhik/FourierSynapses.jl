
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

toBits(a::Number) = a < 0
toBits(a::AbstractArray) = toBits.(a)


# XOR function 

nSynapses = 7
f(a, x) = (1.0./(2.0.^length(a))).*(prod((1 .- (a).*(x))))

a = randn(Float32, nSynapses)
x = rand(Bool, nSynapses) .|> toSpins

f(a, x)

y, back = pullback(f, a, x)
back(1)

dataset = []
# XOR dataset
for i in 0:2^(nSynapses)-1
	p = (bitview(i) |> BitArray)[1:nSynapses] # TODO doesnot handle more than 128 bits
	push!(dataset, [p, xor(p...)])
end

opt = Descent(1e-9)

fPosMean = 0
fNegMean = 0

# This loop repeats 'repeats' times for each data point on average
repeats = 50000
for epoch in 1:(length(dataset)*repeats)
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumSpins = datum |> toSpins
	(y, back) = pullback(a, datumSpins, label) do aa, da, l
		(toSpins(l) - f(aa, da))^2
	end
	if label == 0
		fPosMean = (fPosMean + f(a, datumSpins))/2
	else
		fNegMean = (fPosMean + f(a, datumSpins))/2
	end
	(grads, _) = back(1)
	a .+= toSpins(label).*grads*opt.eta
	# a .-= grads*opt.eta
end

# threshold = (fPosMean + fNegMean)/2

matches = 0
for epoch in 1:(dataset |> length)
	idx = epoch
	(datum, label) = dataset[idx]
	datumSpins = datum |> toSpins
	value = f(a, datumSpins)
	@info value, a, datumSpins, label
	activation = value < 0
	if label == Int(activation)
		matches += 1
	end
end

print(matches/length(dataset))

# plot(alignmentList, color=:lightblue, label="alignment")
# 
# plot!(thresholdList, color=:darkmagenta, label="threshold")
# 
