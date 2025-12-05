"""
Find the membership degree of the number in a fuzzy set defined on 0–10 using a triangular membership function.

Membership Function Definition:
Small Numbers (triangle):
0 → μ = 1
5 → μ = 0.5
10 → μ = 0

"""


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


x = 7
mu = triangular(x, 0, 0, 10)
print("Membership of", x, "=", mu)
