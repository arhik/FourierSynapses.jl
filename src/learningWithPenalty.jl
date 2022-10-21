using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools
using Images
using ImageView
using Plots

a = BitVector(rand(Bool, 2^10))

datasetLength = 400
xdata = unique([BitVector(rand(Bool, 2^10)) for _ in 1:datasetLength])
ydata = rand(Bool, datasetLength)

data = zip(xdata, ydata) 

# c = a * (adjoint(b))

toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1
@inline toSpins(a::Val{false}) = +1
toSpins(a::BitArray) = toSpins.(a)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

using FFTW
using Hadamard

aCoeffs = fwht(a |> toSpins)

w = rand(length(a))

opt = Descent(0.1f0)

image = []
ds = []
overlap = []

for epoch in 1:10000
	idx = rand(1:datasetLength)
	(b, l) = iterate(data, (idx, idx)) |> first
	bCoeffs = fwht(b |> toSpins)
	lSpin = toSpins(l)
	grads = gradient((x, y, l) -> 0.5*(l - 1)*dot(x, y), aCoeffs, bCoeffs, lSpin |> Float64)
	if lSpin == 1
		continue
	end
	aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	aCoeffs .= aCoeffs/norm(aCoeffs)
	push!(image, toBits(aCoeffs))
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
	push!(ds, dot(aCoeffs, bCoeffs))
end

# N Degree convergence
bitImage = hcat(image...);
img = Gray.(0.5.*(toSpins(repeat(bitImage, inner=(10, 1))) .+ 1));
imshow(img);
plot(ds)

# Test
for epoch in 1:10000
	b = BitVector(rand(Bool, 2^10)) 
	while (b in xdata)
		b = BitVector(rand(Bool, 2^10))
	end
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
	idx = rand(1:datasetLength)
	(b, l) = iterate(data, (idx, idx)) |> first
	grads = gradient((x, y, l, z) -> 0.5*(l)*dot((x.*z), y)/length(x), toSpins(a), toSpins(b), toSpins(l), w)
	w .-= grads[4]*opt.eta
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


