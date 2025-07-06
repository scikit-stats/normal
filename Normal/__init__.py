"""
The normal distribution
"""

import numpy as np
from scipy import special, stats
from importlib.metadata import version as get_version


__all__ = ["Normal"]
__version__ = get_version(__package__)


class _Normal:
    r"""Normal distribution with prescribed mean and standard deviation.

    The probability density function of the normal distribution is:

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp {
            \left( -\frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^2 \right)}

    """

    _normalization = 1 / np.sqrt(2 * np.pi)
    _log_normalization = np.log(2 * np.pi) / 2

    @property
    def __make_distribution_version__(self):
        return "1.16.0"

    @property
    def parameters(self):
        return {
            "mu": {"endpoints": (-np.inf, np.inf), "typical": (-1, 1)},
            "sigma": {"endpoints": (0, np.inf), "typical": (0.5, 1.5)},
        }

    @property
    def support(self):
        return {"endpoints": (-np.inf, np.inf)}

    def logpdf(self, x, *, mu, sigma):
        z = (x - mu) / sigma
        return -(self._log_normalization + z**2 / 2) - np.log(sigma)

    def pdf(self, x, *, mu, sigma):
        z = (x - mu) / sigma
        return self._normalization / sigma * np.exp(-(z**2) / 2)

    def logcdf(self, x, *, mu, sigma):
        z = (x - mu) / sigma
        return special.log_ndtr(z)

    def cdf(self, x, *, mu, sigma):
        z = (x - mu) / sigma
        return special.ndtr(z)

    def logccdf(self, x, *, mu, sigma):
        z = (-x - mu) / sigma
        return special.log_ndtr(z)

    def ccdf(self, x, *, mu, sigma):
        z = (-x - mu) / sigma
        return special.ndtr(z)

    def icdf(self, x, *, mu, sigma):
        return special.ndtri(x) * sigma + mu

    def ilogcdf(self, x, *, mu, sigma, **kwargs):
        return special.ndtri_exp(x) * sigma + mu

    def iccdf(self, x, *, mu, sigma):
        return -special.ndtri(x) * sigma + mu

    def ilogccdf(self, x, *, mu, sigma):
        return -special.ndtri_exp(x) * sigma + mu

    def entropy(self, *, mu, sigma):
        return (1 + np.log(2 * np.pi)) / 2 + np.log(abs(sigma))

    def logentropy(self, *, mu, sigma):
        lH0 = np.log1p(np.log(2 * np.pi)) - np.log(2)
        with np.errstate(divide="ignore"):
            # sigma = 1 -> log(sigma) = 0 -> log(log(sigma)) = -inf
            # Silence the unnecessary runtime warning
            lls = np.log(np.log(abs(sigma)) + 0j)
        return special.logsumexp(np.broadcast_arrays(lH0, lls), axis=0)

    def median(self, *, mu, sigma):
        return mu

    def mode(self, *, mu, sigma):
        return mu

    def moment(self, order, kind, mu, sigma):
        if kind == "raw":
            return self._moment_raw(order, mu=mu, sigma=sigma)
        elif kind == "central":
            return self._moment_central(order, mu=mu, sigma=sigma)
        else:
            return None

    def _moment_raw(self, order, *, mu, sigma):
        if order == 0:
            return np.ones_like(mu)
        elif order == 1:
            return mu
        else:
            return None

    def _moment_central(self, order, *, mu, sigma):
        if order == 0:
            return np.ones_like(mu)
        elif order % 2:
            return np.zeros_like(mu)
        else:
            # exact is faster (and obviously more accurate) for reasonable orders
            return sigma**order * special.factorial2(int(order) - 1, exact=True)

    def sample(self, shape, rng, *, mu, sigma):
        return rng.normal(loc=mu, scale=sigma, size=shape)[()]


Normal = stats.make_distribution(_Normal())

# TODO:
# - How to add documentation?
# - Default values of parameters?
# - Generic tests?
# - Typing?
