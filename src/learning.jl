using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using Images
using ImageView
using Plots

using FFTW
using Hadamard

a = BitVector(rand(Bool, 2^10))

dataset = unique([BitVector(rand(Bool, 2^10)) for _ in 1:400])

# c = a * (adjoint(b))

toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1
@inline toSpins(a::Val{false}) = +1
toSpins(a::BitArray) = toSpins.(a)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

aCoeffs = fwht(a |> toSpins)

w = rand(length(a))

opt = Descent(0.1f0)

image = []
ds = []
overlap = []

for epoch in 1:10000
	b = dataset[rand(1:400)]
	bCoeffs = fwht(b |> toSpins)
	grads = gradient((x, y) -> -dot(x, y), aCoeffs, bCoeffs)
	aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	aCoeffs .= aCoeffs/norm(aCoeffs)
	push!(image, toBits(aCoeffs))
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
	push!(ds, dot(aCoeffs, bCoeffs))
end

# N Degree convergence
bitImage = hcat(image...)
img = Gray.(0.5.*(toSpins(repeat(bitImage, inner=(10, 1))) .+ 1));
imshow(img);
plot(ds)

# Test
for epoch in 1:10000
	b = BitVector(rand(Bool, 2^10))
	bCoeffs = fwht(b |> toSpins)
	# grads = gradient((x, y) -> -dot(x, y), aCoeffs, bCoeffs)
	# aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	# aCoeffs .= aCoeffs/norm(aCoeffs)
	# push!(image, toBits(bCoeffs))
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
	push!(ds, dot(aCoeffs, bCoeffs))
end

plot(ds)

# -------------------------------------

image2 = []
ds2 = []
for epoch in 1:10000
	b = dataset[rand(1:10)]
	grads = gradient((x, y, z) -> -dot((x.*z), y)/length(x), toSpins(a), toSpins(b), w)
	w .-= grads[3]*opt.eta
	clamp!(w, 0.0, 1.0)
	push!(ds2, dot(a.*w, b)/length(a))
	push!(image2, a)
end

# Degree 1 convergence

bitImage2 = hcat(image2...)
img2 = Gray.(0.5.*(toSpins(repeat(bitImage2, inner=(10, 1))) .+ 1))

imshow(img2);

plot(ds2)

# Test
for epoch in 1:10000
	b = BitVector(rand(Bool, 2^10))
	# grads = gradient((x, y) -> -dot(x, y), aCoeffs, bCoeffs)
	# aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	# aCoeffs .= aCoeffs/norm(aCoeffs)
	# push!(image, toBits(bCoeffs))
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
	push!(ds2, dot(a.*w, b)/length(a))
end

plot(ds2)

# ------------------------------------

struct Segment{T}
    a::BitVector # TODO keep track of when bit flips
    w::AbstractVector{T}
end

function (s::Segment{T})(x::BitArray) where {T}
    fwht(toSpin.(s.a)) .* s.w
end

function Segment(dims::NTuple{N,T}) where {N,T}
	
end

struct DendriteNode{T}
    a::BitVector
end

function Hadamard.fwht(a::BitArray, dim::Int)
	dims = [size(a)...]
	permuteDims = [1:(dim-1)..., (dim+1:length(dims))..., dim]
	
	ret = Array{Float64}(undef, size(a))
	aPermuted = permutedims
	indxParams = let dim=dim;
		idxParams = []
		for (sidx, s) in enumerate(size(a))
			push!(idxParams, dim==sidx ? Colon() : (1:s))
		end
		idxParams
	end
	idxs = CartesianIndices((indxParams...))
	for idx in idxs
		@info a[idx]		# ret[idx] .= fwht(a[idx])
	end	
end


function Hadamard.fwht(a::BitArray, dim::Int)
	dims = size(a)
	idxParams = [dim==sidx ? sidx : (1:s) for (sidx, s) in enumerate(size(a))]
	ret = Array{Float64}(undef, size(a))
	for idx in eachindex(view(toSpins(a), idxParams...))
		# @info idx.I
		ret[idx.I..., :] .= fwht(a[idx.I..., :] |> toSpins)
	end
	return ret
end


