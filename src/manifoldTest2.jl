
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using BitView
using Images


# Dataset
dataset = []

for i in 1:10000
	p = 2.0*rand(2) .- 1.0
	if norm(p) < 0.5
		push!(dataset, (p, 0))
	else
		push!(dataset, (p, 1))
	end
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
toBits(a::AbstractArray) = toBits.(ifwht(a))

using FFTW
using Hadamard

dendrite = BitArray(undef, 2*(8*sizeof(eltype(dataset[1][1]))))

dendCoeffs = fwht(dendrite |> toSpins)

opt = Descent(0.01f0)

posMean = 0
negMean = 0

for epoch in 1:60000
	idx = rand(1:length(dataset))
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview[:] |> toSpins, 1)
	(y, gradsBack) = pullback(datumCoeffs, dendCoeffs, label) do x, y, l
		alignment = dot(x, y)
		global posMean, negMean
		if l == 1
			negMean = (negMean + alignment)/2
		else
			posMean = (posMean + alignment)/2
		end 
		(l-alignment)^2
	end
	grads = gradsBack(1)
	dendCoeffs .-= grads[2]*opt.eta
	# dendCoeffs[nextpow(2, length(dendCoeffs)/4) : end] .= 0
	dendCoeffs .= dendCoeffs/(norm(dendCoeffs))
end

for epoch in 1:1000
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview |> toSpins, 1)
	@show dot(datumCoeffs, dendCoeffs), label
end

threshold = (posMean + negMean)/2

testresult = []

for epoch in 1:1000
	idx = epoch
	(datum, label) = dataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview[:] |> toSpins, 1)
	push!(testresult, (datum, dot(datumCoeffs, dendCoeffs) > threshold))
end

testdataset = []

for i in 1:10000
	p = 2.0*rand(2) .- 1.0
	if norm(p) < 0.5
		push!(testdataset, (p, 0))
	else
		push!(testdataset, (p, 1))
	end
end

using Plots

scatter(map(x -> first(x) |> first, testdataset), map(x -> first(x) |> last, testdataset), markercolor=map(c -> ifelse(c == 0, :blue, :red), map(last, testdataset)))

testresult = []

for epoch in 1:10000
	idx = epoch
	(datum, label) = testdataset[idx]
	datumview = bitview(datum)
	datumCoeffs = fwht(datumview[:] |> toSpins, 1)
	push!(testresult, (datum, dot(datumCoeffs, dendCoeffs) > threshold))
end

using Plots

scatter(map(x -> first(x) |> first, testresult), map(x -> first(x) |> last, testresult), markercolor=map(c -> ifelse(c == 0, :blue, :red), map(last, testresult)))


