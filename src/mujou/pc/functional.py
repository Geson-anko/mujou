"""The stateless computations of PC-ALM that the nodes of ``mujou.pc`` are made
of: ``vjp`` and ``loss_vjp`` (the gradients of a node's energy), ``relax_`` (the
dynamics of ``u`` and ``lam``) and ``learn_`` (the update of the parameters).

Use them to write a node of your own, such as one with several inputs. Gradients
are summed over the batch; ``learn_`` applies the ``1 / B``.
"""

from mujou._core.pc.functional import learn_, loss_vjp, relax_, vjp

__all__ = ["learn_", "loss_vjp", "relax_", "vjp"]
