import pytest

from windtunnel.config import TunnelSpec
from windtunnel.design import design_tunnel


@pytest.fixture(scope="session")
def spec_m2():
    s = TunnelSpec()
    s.design.mach = 2.0
    s.nozzle.n_axis = 60
    return s


@pytest.fixture(scope="session")
def design_m2(spec_m2):
    return design_tunnel(spec_m2.copy())
