import luminis_mc as lm

function = lm.RayleighPhaseFunction(100, -1.0, 1.0203)
expmaples = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for example in expmaples:
    result = function.sample(example)
    print(f"Phase function value for {example}: {result}")

uniform = lm.UniformPhaseFunction()
for example in expmaples:
    result = uniform.sample(example)
    print(f"Uniform phase function value for {example}: {result}")
