using LinearAlgebra
using Zygote
using Optimisers

a = BitVector(rand(Bool, 2^16))
b = BitVector(rand(Bool, 2^16))

c = a * (adjoint(b))

using FFTW
using Hadamard

frombits(a::BitVector) = begin
    [((bit == true) ? (+1) : (-1)) |> Int8 for bit in a]
end

aCoeffs = fwht(a |> frombits)
bCoeffs = fwht(b |> frombits)

d = dot(aCoeffs, bCoeffs)

gradient(dot, aCoeffs, bCoeffs)

opt = Descent(0.01f0)

@btime for epoch in 1:100
	grads = gradient((x, y) -> -dot(x, y), aCoeffs, bCoeffs)
	aCoeffs .-= grads[1]*opt.eta
	# bCoeffs .-= grads[2]*opt.eta
	aCoeffs .= aCoeffs/norm(aCoeffs)
	# bCoeffs .= bCoeffs/norm(bCoeffs)
	# bCoeffs .-= grads[2]*opt.eta
	@info dot(aCoeffs, bCoeffs)
	@
end

toSpin(a::Bool) = @inline toSpin(Val(a))
@inline toSpin(a::Val{true}) = -1
@inline toSpin(a::Val{false}) = +1

toBits(a::Number) = a > 0
toBits(a::AbstractArray) = toBits.(ifwht(a))

# broadcast(toSpin, a)

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
