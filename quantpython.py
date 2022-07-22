import QuantLib as ql


unifMt = ql.MersenneTwisterUniformRng()
bmGauss = ql.BoxMullerMersenneTwisterGaussianRng(unifMt)

for i in range(5):
    print(bmGauss.next().value())

