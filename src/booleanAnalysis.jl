
using LinearAlgebra
using Zygote
using Optimisers
using BenchmarkTools

a = BitVector(rand(Bool, 2^4))
b = BitVector(rand(Bool, 2^4))

dataset = unique([BitVector(rand(Bool, 2^4)) for _ in 1:2^4])

c = a * (adjoint(b))

toSpins(a::Bool) = @inline toSpins(Val(a))
@inline toSpins(a::Val{true}) = -1
@inline toSpins(a::Val{false}) = +1
toSpins(a::BitVector) = toSpins.(a)

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

using FFTW
using Hadamard

aCoeffs = fwht(a |> toSpins)
bCoeffs = fwht(b |> toSpins)

d = dot(aCoeffs, bCoeffs)

gradient(dot, aCoeffs, bCoeffs)

opt = Descent(0.01f0)

for epoch in 1:100
	grads = gradient((x, y) -> -dot(x, y), aCoeffs, bCoeffs)
	aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	aCoeffs .= aCoeffs/(norm(aCoeffs, 1) + 1e-5)
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
end

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

