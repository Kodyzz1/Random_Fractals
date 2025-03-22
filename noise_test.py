import opensimplex

opensimplex.seed(42)
noise_value = opensimplex.noise2(1.0, 2.0)
print(noise_value)