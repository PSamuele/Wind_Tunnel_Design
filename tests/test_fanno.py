import pytest

from windtunnel import fanno as fn


def test_fanno_tables():
    # NACA 1135 / standard Fanno tables, gamma = 1.4
    assert fn.fanno_fL_D(2.0) == pytest.approx(0.30500, rel=2e-4)
    assert fn.fanno_p_pstar(2.0) == pytest.approx(0.40825, rel=1e-4)
    assert fn.fanno_p0_p0star(2.0) == pytest.approx(1.6875, rel=1e-4)
    assert fn.fanno_fL_D(0.5) == pytest.approx(1.06906, rel=2e-4)


def test_fanno_exit_mach_and_choking():
    M_out, choked = fn.fanno_exit_mach(2.0, 0.1)
    assert not choked and 1.0 < M_out < 2.0
    assert fn.fanno_length_to_mach(2.0, M_out) == pytest.approx(0.1, rel=1e-9)
    M_out, choked = fn.fanno_exit_mach(0.5, 0.5)
    assert not choked and 0.5 < M_out < 1.0
    assert fn.fanno_exit_mach(2.0, 0.5) == (1.0, True)


def test_friction_factor():
    assert fn.darcy_friction_factor(1000.0) == pytest.approx(0.064)
    # smooth pipe, Re = 1e5: Colebrook gives about 0.0180
    assert fn.darcy_friction_factor(1e5) == pytest.approx(0.0180, rel=0.03)
