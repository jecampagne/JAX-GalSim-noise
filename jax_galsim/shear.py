import jax.numpy as jnp


import galsim as _galsim
from jax._src.numpy.util import _wraps

@_wraps(
    _galsim.Shear,
    lax_description="Does not support (yet) all shear parametrisation schema",
)
class Shear():
    def __init__(self, *args, **kwargs):

        # There is no valid set of >2 keyword arguments, so raise an exception in this case:
        if len(kwargs) > 2:
            raise TypeError(
                "Shear constructor received >2 keyword arguments: %s"%kwargs.keys())

        if len(args) > 1:
            raise TypeError(
                "Shear constructor received >1 non-keyword arguments: %s"%str(args))

        # If a component of e, g, or eta, then require that the other component is zero if not set,
        # and don't allow specification of mixed pairs like e1 and g2.
        # Also, require a position angle if we didn't get g1/g2, e1/e2, or eta1/eta2

        # Unnamed arg must be a complex shear
        if len(args) == 1:
            self._g = args[0]
            if not isinstance(self._g, complex):
                raise TypeError("Non-keyword argument to Shear must be complex g1 + 1j * g2")

        # Empty constructor means shear == (0,0)
        elif not kwargs:
            self._g = 0j

        # g1,g2
        elif 'g1' in kwargs or 'g2' in kwargs:
            g1 = kwargs.pop('g1', 0.)
            g2 = kwargs.pop('g2', 0.)
            self._g = g1 + 1j * g2
            if abs(self._g) > 1.:
                raise _galsim.GalSimRangeError("Requested shear exceeds 1.", self._g, 0., 1.)

        # e1,e2
        elif 'e1' in kwargs or 'e2' in kwargs:
            e1 = kwargs.pop('e1', 0.)
            e2 = kwargs.pop('e2', 0.)
            absesq = e1**2 + e2**2
            if absesq > 1.:
                raise _galsim.GalSimRangeError("Requested distortion exceeds 1.",np.sqrt(absesq), 0., 1.)
            self._g = (e1 + 1j * e2) * self._e2g(absesq)

        # eta1,eta2
        elif 'eta1' in kwargs or 'eta2' in kwargs:
            eta1 = kwargs.pop('eta1', 0.)
            eta2 = kwargs.pop('eta2', 0.)
            eta = eta1 + 1j * eta2
            abseta = jnp.abs(eta)
            self._g = eta * self._eta2g(abseta)

        # g,beta
        elif 'g' in kwargs:
            raise NotImplementedError("Shear given 'g' and 'beta' not yet Implemented")
##             if 'beta' not in kwargs:
##                 raise _galsim.GalSimIncompatibleValuesError(
##                     "Shear constructor requires beta when g is specified.",
##                     g=kwargs['g'], beta=None)
##             beta = kwargs.pop('beta')
##             if not isinstance(beta, Angle):
##                 raise TypeError("beta must be an Angle instance.")
##             g = kwargs.pop('g')
##             if g > 1 or g < 0:
##                 raise GalSimRangeError("Requested |shear| is outside [0,1].",g, 0., 1.)
##             self._g = g * np.exp(2j * beta.rad)

        # e,beta
        elif 'e' in kwargs:
            raise NotImplementedError("Shear given 'e' and 'beta' not yet Implemented")

##             if 'beta' not in kwargs:
##                 raise GalSimIncompatibleValuesError(
##                     "Shear constructor requires beta when e is specified.",
##                     e=kwargs['e'], beta=None)
##             beta = kwargs.pop('beta')
##             if not isinstance(beta, Angle):
##                 raise TypeError("beta must be an Angle instance.")
##             e = kwargs.pop('e')
##             if e > 1 or e < 0:
##                 raise GalSimRangeError("Requested distortion is outside [0,1].", e, 0., 1.)
##             self._g = self._e2g(e**2) * e * np.exp(2j * beta.rad)

        # eta,beta
        elif 'eta' in kwargs:
            raise NotImplementedError("Shear given 'eta' and 'beta' not yet Implemented")
##             if 'beta' not in kwargs:
##                 raise GalSimIncompatibleValuesError(
##                     "Shear constructor requires beta when eta is specified.",
##                     eta=kwargs['eta'], beta=None)
##             beta = kwargs.pop('beta')
##             if not isinstance(beta, Angle):
##                 raise TypeError("beta must be an Angle instance.")
##             eta = kwargs.pop('eta')
##             if eta < 0:
##                 raise GalSimRangeError("Requested eta is below 0.", eta, 0.)
##             self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad)

        # q,beta
        elif 'q' in kwargs:
            raise NotImplementedError("Shear given 'q' and 'beta' not yet Implemented")

##             if 'beta' not in kwargs:
##                 raise GalSimIncompatibleValuesError(
##                     "Shear constructor requires beta when q is specified.",
##                     q=kwargs['q'], beta=None)
##             beta = kwargs.pop('beta')
##             if not isinstance(beta, Angle):
##                 raise TypeError("beta must be an Angle instance.")
##             q = kwargs.pop('q')
##             if q <= 0 or q > 1:
##                 raise GalSimRangeError("Cannot use requested axis ratio.", q, 0., 1.)
##             eta = -np.log(q)
##             self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad)

        elif 'beta' in kwargs:
            raise GalSimIncompatibleValuesError(
                "beta provided to Shear constructor, but not g/e/eta/q",
                beta=kwargs['beta'], e=None, g=None, q=None, eta=None)

        # check for the case where there are 1 or 2 kwargs that are not valid ones for
        # initializing a Shear
        if kwargs:
            raise TypeError(
                "Shear constructor got unexpected extra argument(s): %s"%kwargs.keys())

    @property
    def g1(self):
        """The first component of the shear in the "reduced shear" definition.
        """
        return self._g.real

    @property
    def g2(self):
        """The second component of the shear in the "reduced shear" definition.
        """
        return self._g.imag

    @property
    def g(self):
        """The magnitude of the shear in the "reduced shear" definition.
        """
        return jnp.abs(self._g)

    def _g2e(self, absgsq):
        return 2. / (1.+absgsq)


    def _e2g(self, absesq):
        if absesq > 1.e-4:
            #return (1. - np.sqrt(1.-absesq)) / absesq
            return 1. / (1. + np.sqrt(1.-absesq))
        else:
            # Avoid numerical issues near e=0 using Taylor expansion
            return 0.5 + absesq*(0.125 + absesq*(0.0625 + absesq*0.0390625))

    def _g2eta(self, absg):
        if absg > 1.e-4:
            return 2.*jnp.arctanh(absg)/absg
        else:
            # This doesn't have as much trouble with accuracy, but have to avoid absg=0,
            # so might as well Taylor expand for small values.
            absgsq = absg * absg
            return 2. + absgsq*((2./3.) + absgsq*0.4)

    def _eta2g(self, abseta):
        if abseta > 1.e-4:
            return jnp.tanh(0.5*abseta)/abseta
        else:
            absetasq = abseta * abseta
            return 0.5 + absetasq*((-1./24.) + absetasq*(1./240.))

    # define all the various operators on Shear objects
    def __neg__(self):
        return _Shear(-self._g)

    # order of operations: shear by other._shear, then by self._shear
    def __add__(self, other):
        return _Shear((self._g + other._g) / (1. + self._g.conjugate() * other._g))

    # order of operations: shear by -other._shear, then by self._shear
    def __sub__(self, other):
        return self + (-other)

    def __eq__(self, other):
        return self is other or (isinstance(other, Shear) and self._g == other._g)
    def __ne__(self, other):
        return not self.__eq__(other)



    @_wraps(_galsim.Shear.getMatrix)
    def getMatrix(self):
        return jnp.array([[ 1.+self.g1,  self.g2   ],
                         [  self.g2  , 1.-self.g1 ]]) / jnp.sqrt(1.-self.g**2)

    @_wraps(_galsim.Shear.rotationWith)
    def rotationWith(self, other):
        S3 = self.getMatrix().dot(other.getMatrix()[:,:1])
        R = (-(self + other)).getMatrix().dot(S3)
        theta = jnp.arctan2(R[1,0], R[0,0])
        return theta * radians

    def _Shear(g):
        """Equivalent to ``galsim.Shear(g=g)``, but without the overhead of the normal sanity checks
        and other options for how to specify the shear.

        Parameters:
            g =  g1 + 1j * g2.

        Returns:
            a `galsim.Shear` instance
        """
        return Shear(g=g)
