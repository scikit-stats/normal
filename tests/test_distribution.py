import numpy as np

from normal import Normal


def _pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def test_pdf():
    rng = np.random.default_rng(594875248529782456)
    mu = rng.normal(size=10)
    sigma = rng.random(size=10)
    x = rng.random(size=(10, 1))
    X = Normal(mu=mu, sigma=sigma)
    res = X.pdf(x)
    ref = _pdf(x, mu, sigma)
    np.testing.assert_allclose(res, ref)
