"""Assignment 1, Questions 1-6. Standard library only; Python 3.12."""
from fractions import Fraction as F
from math import isclose, log2


def solve():
    a, b = F(2, 5), F(3, 10)
    intersection = a * b
    union = a + b - intersection
    print(f"Q1(a): P(A and B) = 0.4 * 0.3 = {float(intersection):.2f}")
    print(f"Q1(b): P(A or B) = 0.4 + 0.3 - 0.12 = {float(union):.2f}")
    print("Q2: Not independent: P(A|B) = 0.7 != P(A) = 0.5.")
    print("    P(A and B) = 0.7 * 0.4 = 0.28 != 0.5 * 0.4 = 0.20.")
    marginal = F(1, 2) * F(3, 5) + F(1, 5) * F(2, 5)
    posterior = F(1, 2) * F(3, 5) / marginal
    print(f"Q3: P(B) = 0.5*0.6 + 0.2*0.4 = {float(marginal):.2f}")
    print(f"    P(A|B) = 0.30/0.38 = {posterior} = {float(posterior):.6f}")
    positive = F(95, 100) * F(2, 100) + F(10, 100) * F(98, 100)
    disease_given_positive = F(95, 100) * F(2, 100) / positive
    print(f"Q4: P(+) = 0.95*0.02 + 0.10*0.98 = {float(positive):.3f}")
    print(f"    P(D|+) = 0.019/0.117 = {disease_given_positive} = {float(disease_given_positive):.6f} ({float(disease_given_positive)*100:.4f}%)")
    scores = [85, 90, 95, 100]
    probabilities = [F(3, 8), F(3, 8), F(1, 8), F(1, 8)]
    mean = sum(x*p for x, p in zip(scores, probabilities))
    second = sum(x*x*p for x, p in zip(scores, probabilities))
    variance = second - mean*mean
    sample = [85, 90, 85, 95, 90, 85, 100, 90]
    sample_mean = F(sum(sample), len(sample))
    print(f"Q5(a): E[X] = {float(mean):.2f}")
    print(f"Q5(b): E[X^2] = {float(second):.2f}; Var(X) = {float(variance):.4f}")
    print(f"Q5(c): Sample mean = {sum(sample)}/8 = {float(sample_mean):.2f}; equals E[X].")
    entropy = -sum(p * log2(p) for p in [0.4, 0.3, 0.2, 0.1])
    uniform_entropy = -4 * 0.25 * log2(0.25)
    print(f"Q6(a): H(X) = {entropy:.8f} bits")
    print(f"Q6(b): H(uniform) = {uniform_entropy:.0f} bits")
    assert intersection == F(3, 25) and union == F(29, 50)
    assert posterior == F(15, 19) and disease_given_positive == F(19, 117)
    assert mean == sample_mean == F(720, 8)
    # Independent definition of variance verifies the second-moment calculation.
    assert variance == sum(p * (x - mean)**2 for x, p in zip(scores, probabilities))
    assert isclose(entropy, 1.8464393446710154)
    assert uniform_entropy == 2


if __name__ == "__main__":
    solve()
