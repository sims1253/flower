import torch

from flower.flows.coupling import ConditionalCouplingFlow


def test_conditional_coupling_inverse_reconstructs_input():
    flow = ConditionalCouplingFlow(dim=16, cond_dim=8, layers=3)
    z = torch.randn(4, 16)
    cond = torch.randn(4, 8)
    zp = flow(z, cond)
    reconstructed = flow.inverse(zp, cond)
    assert torch.allclose(reconstructed, z, atol=1e-5)
