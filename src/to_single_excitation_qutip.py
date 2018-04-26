"""Conversion of QNET expressions to qutip objects in the single-excitation
subspace"""
from copy import copy
from scipy.sparse import csr_matrix
from qnet.algebra.abstract_algebra import AlgebraError
from qnet.algebra.circuit_algebra import SLH
from qnet.algebra.operator_algebra import (
        IdentityOperator, ZeroOperator, Create, Destroy, Jz,
        Jplus, Jminus, Phase, Displace, Squeeze, LocalSigma, OperatorOperation,
        OperatorPlus, OperatorTimes, ScalarTimesOperator, Adjoint,
        PseudoInverse, OperatorTrace, NullSpaceProjector, Operation, Operator,
        LocalProjector, LocalOperator)
from qnet.algebra.state_algebra import (
        Ket, BraKet, KetBra, BasisKet, CoherentStateKet, KetPlus, TensorKet,
        ScalarTimesKet, OperatorTimesKet)
from qnet.algebra.hilbert_space_algebra import (
        TrivialSpace, LocalSpace, ProductSpace)
from qnet.algebra.super_operator_algebra import (
        SuperOperator, IdentitySuperOperator, SuperOperatorPlus,
        SuperOperatorTimes, ScalarTimesSuperOperator, SPre, SPost,
        SuperOperatorTimesOperator, ZeroSuperOperator)
from qnet.algebra.pattern_matching import pattern, wc
from bitarray import bitarray

import qutip


__all__ = ['convert_to_single_excitation_qutip']


def convert_to_single_excitation_qutip(
        expr, bit_index, full_space=None, mapping=None):
    """Convert a QNET expression to a qutip object

    Args:
        expr: a QNET expression
        bit_index: a dictionary that maps spin configuration (as 01-strings)
            to a 0-based index in the encoded Hilbert space
        full_space (HilbertSpace): The
            Hilbert space in which `expr` is defined. If not given,
            ``expr.space`` is used. The Hilbert space must have a well-defined
            basis.
        mapping (dict): A mapping of any (sub-)expression to either a
            `quip.Qobj` directly, or to a callable that will convert the
            expression into a `qutip.Qobj`. Useful for e.g. supplying objects
            for symbols
    Raises:
        ValueError: if `expr` is not in `full_space`, or if `expr` cannot be
            converted.
    """
    if full_space is None:
        full_space = expr.space
    qutip_dimension = len(full_space.local_factors)
    for space in full_space.local_factors:
        assert space.dimension == 2

    if not expr.space.is_tensor_factor_of(full_space):
        raise ValueError(
            "expr '%s' must be in full_space %s" % (expr, full_space))
    if full_space == TrivialSpace:
        raise AlgebraError(
            "Cannot convert object in TrivialSpace to qutip. "
            "You may pass a non-trivial `full_space`")
    if mapping is not None:
        raise NotImplementedError()
    if expr is IdentityOperator:
        local_spaces = full_space.local_factors
        if len(local_spaces) == 0:
            raise ValueError("full_space %s does not have local factors"
                             % full_space)
        else:
            return qutip.qeye(qutip_dimension)
    elif expr is ZeroOperator:
        return qutip.Qobj(csr_matrix((qutip_dimension, qutip_dimension)))
    elif isinstance(expr, LocalOperator):
        return _convert_local_operator_to_qutip(
            expr, bit_index, full_space, mapping)
    elif isinstance(expr, OperatorOperation):
        return _convert_operator_operation_to_qutip(
            expr, bit_index, full_space, mapping)
    elif isinstance(expr, ScalarTimesOperator):
        try:
            coeff = complex(expr.coeff)
        except TypeError:
            raise TypeError("Scalar coefficient '%s' is not numerical" %
                            expr.coeff)
        return (
            coeff *
            convert_to_single_excitation_qutip(
                expr.term, bit_index, full_space=full_space, mapping=mapping))
    elif isinstance(expr, SLH):
        # SLH object cannot be converted to a single qutip object, only to a
        # nested list of qutip object. That's why a separate routine
        # SLH_to_qutip exists
        raise ValueError("SLH objects can only be converted using "
                         "SLH_to_qutip routine")
    else:
        raise ValueError("Cannot convert '%s' of type %s"
                         % (str(expr), type(expr)))


def construct_bit_index(full_space):
    """Return the bit-index for the zero/single-excitation subspace"""
    N = len(full_space.local_factors)
    zero = bitarray(N)
    zero.setall(0)
    bit_index = {zero.to01(): 0}  # zero excitation subspace
    for (i, hs) in enumerate(reversed(full_space.local_factors)):
        assert hs.dimension == 2
        key = copy(zero)
        key[N-i-1] = 1
        bit_index[key.to01()] = i+1
    return bit_index


def bit_key(local_space, full_space):
    """Return the lookup key in the bit-index for the given local space"""
    assert isinstance(full_space, ProductSpace)
    assert isinstance(local_space, LocalSpace)
    key = bitarray(len(full_space.local_factors))
    key.setall(0)
    key[full_space.operands.index(local_space)] = 1
    return key


_PAT_ADAG_A = pattern(
    OperatorTimes,
    wc('adag', head=Create, kwargs={
        'hs': wc('hs1', head=LocalSpace)}),
    wc('a', head=Destroy, kwargs={
        'hs': wc('hs2', head=LocalSpace)}))

_PAT_A_ADAG = pattern(
    OperatorTimes,
    wc('a', head=Destroy, kwargs={
        'hs': wc('hs1', head=LocalSpace)}),
    wc('adag', head=Create, kwargs={
        'hs': wc('hs2', head=LocalSpace)}))

_PAT_SHIFT_OP = pattern(
    OperatorTimes,
    pattern(LocalProjector, 'g'),
    wc('adag', head=Create, kwargs={
        'hs': wc('hs1', head=LocalSpace)}),
    wc('a', head=Destroy, kwargs={
        'hs': wc('hs2', head=LocalSpace)}))


def _qutip_sigma(hs1, hs2, full_space, bit_index):
    N = len(bit_index)
    key1 = bit_key(hs1, full_space).to01()
    key2 = bit_key(hs2, full_space).to01()
    res = (
        qutip.basis(N, bit_index[key1]) *
        qutip.basis(N, bit_index[key2]).dag())
    return res


def _convert_operator_operation_to_qutip(expr, bit_index, full_space, mapping):
    if isinstance(expr, OperatorPlus):
        res = 0
        for op in expr.operands:
            res += convert_to_single_excitation_qutip(
                op, bit_index, full_space, mapping=mapping)
        return res
    elif isinstance(expr, OperatorTimes):
        m = _PAT_ADAG_A.match(expr)
        if m:
            return _qutip_sigma(m['hs1'], m['hs2'], full_space, bit_index)
        m = _PAT_A_ADAG.match(expr)
        if m:
            return _qutip_sigma(m['hs2'], m['hs1'], full_space, bit_index)
        m = _PAT_SHIFT_OP.match(expr)
        if m:
            return _qutip_sigma(m['hs1'], m['hs2'], full_space, bit_index)
        raise NotImplementedError(
            'Cannot convert operators other than â^† â: %s' % expr)
    elif isinstance(expr, Adjoint):
        return convert_to_single_excitation_qutip(
            qutip.dag(expr.operands[0]), bit_index, full_space,
            mapping=mapping)
    else:
        raise ValueError("Cannot convert '%s' of type %s"
                         % (str(expr), type(expr)))


def _convert_local_operator_to_qutip(expr, bit_index, full_space, mapping):
    if isinstance(expr, Destroy):
        N = len(bit_index)
        key = bit_key(expr.space, full_space).to01()
        return qutip.basis(N, 0) * qutip.basis(N, bit_index[key]).dag()
    else:
        raise ValueError("Cannot convert '%s' of type %s"
                         % (str(expr), type(expr)))
