"""Golden-value tests for coastal.flow deformation metrics.

Pins the flow-deformation math (divergence / vorticity / strain) to known analytic
fields. These would fail under the pre-fix swapped-gradient-axes bug (divergence and
vorticity were computed from the wrong partials). See docs/SEGMENTATION.md appendix.

Convention (matches flow.py): image array axes [y=0, x=1]; u = x-displacement,
v = y-displacement; so ∂/∂x = np.gradient(·, axis=1), ∂/∂y = np.gradient(·, axis=0).
"""

import numpy as np

from coastal.flow import _flow_deformation


def test_flow_deformation_pure_divergence():
    # Radial expansion u = x, v = y  →  divergence = ∂u/∂x + ∂v/∂y = 2, vorticity = 0.
    n = 8
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    div, vort, strain = _flow_deformation(u=xs.copy(), v=ys.copy())
    assert np.allclose(div, 2.0)
    assert np.allclose(vort, 0.0)
    # strain = sqrt(E_xx^2 + E_yy^2 + 2 E_xy^2) = sqrt(1 + 1 + 0) = sqrt(2).
    assert np.allclose(strain, np.sqrt(2.0))


def test_flow_deformation_pure_rotation():
    # Solid-body rotation u = -y, v = x  →  divergence = 0, vorticity = ∂v/∂x - ∂u/∂y = 2.
    n = 8
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
    div, vort, strain = _flow_deformation(u=-ys, v=xs.copy())
    assert np.allclose(div, 0.0)
    assert np.allclose(vort, 2.0)
    # Pure rotation has zero strain rate (antisymmetric velocity gradient).
    assert np.allclose(strain, 0.0)
