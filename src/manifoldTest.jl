
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView


# Dataset
dataset = []

for i in 1:1000
	p = rand()
	if p^2 < 0.5
		push!(dataset, (p, 0))
	else
		push!(dataset, (p, 1))
	end
end

# Utility functions
toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1
@inline toSpins(a::Val{false}) = +1
toSpins(a::BitVector) = toSpins.(a)
toSpins(a::BitMatrix) = toSpins.(a)
toSpins(b::BitViewArray) = toSpins.(b)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

using FFTW
using Hadamard

dendrite = BitArray(undef, (1, 64))

dendCoeffs = fwht(dendrite |> toSpins)

opt = Descent(0.001f0)

for epoch in 1:10000
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	grads = gradient((x, y, l) -> (l-dot(x, y))^2, datumCoeffs, dendCoeffs, label)
	dendCoeffs .-= grads[2]*opt.eta
	dendCoeffs .= dendCoeffs/(norm(dendCoeffs))
end

for epoch in 1:1000
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, ndims(datumview))
	@show dot(datumCoeffs, dendCoeffs), label
	# grads = gradient((x, y) -> (1-dot(x, y))^2, datumCoeffs, dendCoeffs)
	# dendCoeffs .-= grads[2]*opt.eta
	# dendCoeffs .= dendCoeffs/(norm(dendCoeffs))
end



