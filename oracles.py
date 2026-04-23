import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class NonConvexOracle(BaseSmoothOracle):
    """
    Oracle for test function from your assignment.
    """

    def __init__(self):
        pass

    def func(self, x):
        return np.sum(0.25 * x ** 4 - 0.5 * x ** 2)

    def grad(self, x):
        return x ** 3 - x

    def hess(self, x):
        return np.diag(3.0 * x ** 2 - 1.0)

    def hess_vec(self, x, v):
        return (3.0 * x ** 2 - 1.0) * v


class REG_MODEL_NAMEL2Oracle(BaseSmoothOracle):
    """
    Oracle for regression loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax_minus_b = self.matvec_Ax(x) - self.b
        m = self.b.size
        return 0.5 / m * np.dot(Ax_minus_b, Ax_minus_b) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        Ax_minus_b = self.matvec_Ax(x) - self.b
        m = self.b.size
        return self.matvec_ATx(Ax_minus_b) / m + self.regcoef * x

    def hess(self, x):
        m = self.b.size
        n = x.size
        return self.matmat_ATsA(np.ones(m)) / m + self.regcoef * np.eye(n)

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v.
        """
        m = self.b.size
        return self.matvec_ATx(self.matvec_Ax(v)) / m + self.regcoef * v


class CLASS_MODEL_NAMEL2Oracle(BaseSmoothOracle):
    """
    Oracle for classification loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        z = self.b * self.matvec_Ax(x)
        m = self.b.size
        return np.mean(np.logaddexp(0.0, -z)) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = self.b * self.matvec_Ax(x)
        m = self.b.size
        sigma = expit(-z)
        return -self.matvec_ATx(self.b * sigma) / m + self.regcoef * x

    def hess(self, x):
        z = self.b * self.matvec_Ax(x)
        m = self.b.size
        sigma = expit(-z)
        s = sigma * (1.0 - sigma)
        n = x.size
        return self.matmat_ATsA(s) / m + self.regcoef * np.eye(n)

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v.
        """
        z = self.b * self.matvec_Ax(x)
        m = self.b.size
        sigma = expit(-z)
        s = sigma * (1.0 - sigma)
        Av = self.matvec_Ax(v)
        return self.matvec_ATx(s * Av) / m + self.regcoef * v


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    n = x.size
    result = np.zeros_like(x, dtype=float)
    fx = func(x)
    fx_ev = func(x + eps * v)
    for i in range(n):
        e_i = np.zeros_like(x, dtype=float)
        e_i[i] = 1.0
        result[i] = (func(x + eps * v + eps * e_i) - fx_ev - func(x + eps * e_i) + fx) / (eps ** 2)
    return result
