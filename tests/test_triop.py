"""Headless tests for triop (plan §8.3).

Run:  ~/.local/share/triop/.venv/bin/python -m pytest tests/ -q
Unit tests drive sample_system() against synthetic /proc trees (monkeypatch-free:
the sampler takes proc root + clock as parameters). The pilot test boots the real
TUI headlessly via textual.run_test against the live /proc.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triop
from triop import (
    DASH,
    SamplerState,
    _pick_vram,
    cpu_busy_pct,
    fmt_mem,
    read_cpu_total,
    read_gpu,
    sample_system,
)

CLK_TCK = os.sysconf("SC_CLK_TCK")


def stat_line(pid: int, comm: str, utime: int, stime: int, uid: int = 1000, state: str = "S") -> str:
    """Build /proc/<pid>/stat text. Tokens tras ')': [0]=state(field3),
    [11]=utime (field14), [12]=stime (field15), [19]=uid (field22).
    El bloque numérico empieza una posición DESPUÉS de state."""
    toks = ["0"] * 40
    toks[10], toks[11], toks[18] = str(utime), str(stime), str(uid)
    return f"{pid} ({comm}) {state} " + " ".join(toks)


FDINFO = (
    "pos:\t0\nflags:\t02100002\nmnt_id:\t28\ndrm-driver:\ti915\n"
    "drm-client-id:\t26\ndrm-pdev:\t0000:00:02.0\n"
    "drm-total-system0:\t{total} KiB\n"
    "drm-active-system0:\t0\ndrm-resident-system0:\t848\n"
    "drm-engine-render:\t{render} ns\ndrm-engine-copy:\t0 ns\n"
)


class FakeProc:
    """Synthetic /proc tree rooted at a tmp_path."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "meminfo").write_text(
            "MemTotal:  16000000 kB\nMemFree:   4000000 kB\nMemAvailable: 10000000 kB\n"
        )
        self.set_cpu([10, 0, 10, 970, 10, 0, 0, 0, 0, 0])

    def set_cpu(self, fields: list[int]) -> None:
        (self.root / "stat").write_text(
            "cpu  " + " ".join(map(str, fields)) + "\ncpu0  0 0 0 0 0 0 0 0 0 0\n"
        )

    def pid(
        self,
        n: int,
        comm: str,
        utime: int = 0,
        stime: int = 0,
        uid: int = 1000,
        rss_pages: int = 50,
        fdinfos: tuple[str, ...] = (),
    ) -> Path:
        d = self.root / str(n)
        d.mkdir(exist_ok=True)
        (d / "stat").write_text(stat_line(n, comm, utime, stime, uid))
        (d / "statm").write_text(f"123 {rss_pages} 20 5 0 0 0\n")
        fd = d / "fdinfo"
        fd.mkdir(exist_ok=True)
        for i, txt in enumerate(fdinfos):
            (fd / str(i)).write_text(txt)
        return d


@pytest.fixture()
def fake(tmp_path: Path) -> FakeProc:
    return FakeProc(tmp_path)


# ---------------------------------------------------------------- pure helpers


def test_cpu_busy_pct_formula():
    prev = (980, 1000)  # idle_all, total_all
    cur = (1480, 1900)
    assert cpu_busy_pct(prev, cur) == pytest.approx(100 * (1 - 500 / 900), abs=1e-6)


def test_cpu_busy_pct_guards():
    assert cpu_busy_pct((10, 10), (5, 5)) is None      # dt <= 0
    assert cpu_busy_pct((100, 200), (50, 300)) is None # idle went backwards


def test_read_cpu_total_merges_iowait(fake: FakeProc):
    fake.set_cpu([10, 0, 10, 970, 10, 0, 0, 0, 0, 0])
    assert read_cpu_total(fake.root) == (980, 1000)    # idle(970)+iowait(10)
    fake.set_cpu([210, 0, 210, 1470, 10, 0, 0, 0, 0, 0])
    assert read_cpu_total(fake.root) == (1480, 1900)


def test_pick_vram_prefers_system0_and_skips_stolen():
    assert _pick_vram({"system0": 10, "stolen-system0": 999}) == 10
    assert _pick_vram({"video-total": 7, "stolen-system0": 999}) == 7
    assert _pick_vram({"stolen-only": 5}) == 0
    assert _pick_vram({}) == 0


def test_fmt_mem_boundaries():
    assert fmt_mem(None) == DASH
    assert fmt_mem(384 * 1024) == "384.0 KiB"
    assert fmt_mem(512 * 1024**2) == "512.0 MiB"
    assert fmt_mem(int(1.5 * 1024**3)) == "1.5 GiB"
    assert fmt_mem(999) == "999 B"


def test_read_gpu_sums_engines_across_fds_takes_best_vram(tmp_path: Path):
    d = tmp_path / "42"
    fd = d / "fdinfo"
    fd.mkdir(parents=True)
    (fd / "0").write_text(FDINFO.format(total=100, render=10))
    (fd / "1").write_text(
        FDINFO.format(total=60, render=7).replace("drm-total-system0:", "drm-total-video0:")
        + "drm-total-stolen-system0:\t999 KiB\n"
    )
    gi = read_gpu(d)
    assert gi is not None
    assert gi.engines_ns == {"render": 17, "copy": 0}
    assert gi.vram_kib == 100


def test_read_gpu_none_without_drm(tmp_path: Path):
    d = tmp_path / "7"
    fd = d / "fdinfo"
    fd.mkdir(parents=True)
    (fd / "0").write_text("pos:\t0\nflags:\t0100000\n")
    assert read_gpu(d) is None


# ------------------------------------------------------------------- sampler


def two_state_fixture(fake: FakeProc) -> None:
    """Seed fake with pid 42 (GPU worker) and pid 99 (no GPU)."""
    fake.pid(42, "renderd", utime=60, stime=40, rss_pages=50,
             fdinfos=(FDINFO.format(total=384, render=1_000_000),))
    fake.pid(99, "nogpu", utime=0, stime=0)


def advance(fake: FakeProc) -> None:
    fake.set_cpu([210, 0, 210, 1470, 10, 0, 0, 0, 0, 0])
    d = fake.root / "42"
    (d / "stat").write_text(stat_line(42, "renderd", 160, 140))
    (d / "fdinfo" / "0").write_text(FDINFO.format(total=384, render=51_000_000))


def test_sample_two_cycles_percentages_and_totals(fake: FakeProc):
    two_state_fixture(fake)
    st = SamplerState()
    s0 = sample_system(st, proc=fake.root, now=100.0)
    assert s0.totals.cpu_pct is None              # primera muestra sin delta
    p0 = next(p for p in s0.procs if p.pid == 42)
    assert p0.cpu_pct is None and p0.gpu_pct is None
    assert p0.vram == 384 * 1024                  # VRAM visible ya en la 1ª muestra

    advance(fake)
    s1 = sample_system(st, proc=fake.root, now=101.0)
    p42 = next(p for p in s1.procs if p.pid == 42)
    p99 = next(p for p in s1.procs if p.pid == 99)

    exp_cpu = 100 * (200 / CLK_TCK) / 1.0
    assert p42.cpu_pct == pytest.approx(exp_cpu, rel=1e-3)
    assert p42.gpu_pct == pytest.approx(5.0, abs=1e-6)   # 50 ms de 1000 ms
    assert p42.vram == 384 * 1024
    assert p99.cpu_pct == pytest.approx(0.0, abs=1e-9)   # delta 0 -> 0.0
    assert p99.gpu_pct is None                            # sin uso GPU -> '–'
    assert p99.vram is None                               # sin fdinfo DRM -> '–'

    t = s1.totals
    assert t.cpu_pct == pytest.approx(100 * (1 - 500 / 900), abs=1e-3)
    assert t.gpu_busy_pct == pytest.approx(5.0, abs=1e-6)
    assert t.gpu_mem_used == 384 * 1024
    assert t.ram_total == 16_000_000 * 1024
    assert t.ram_used == 6_000_000 * 1024                # MemTotal - MemAvailable
    assert p42.user == "denisio" or p42.user == "1000"   # uid sintético


def test_vanished_pid_does_not_crash(fake: FakeProc):
    two_state_fixture(fake)
    fake.pid(77, "ghost", utime=5, stime=5)
    st = SamplerState()
    sample_system(st, proc=fake.root, now=100.0)
    import shutil

    shutil.rmtree(fake.root / "77")   # el proceso muere entre muestras
    advance(fake)
    s1 = sample_system(st, proc=fake.root, now=101.0)    # no debe lanzar
    assert 77 not in [p.pid for p in s1.procs]
    assert 42 in [p.pid for p in s1.procs]


def test_unreadable_fdinfo_is_dash_not_crash(fake: FakeProc):
    two_state_fixture(fake)
    st = SamplerState()
    sample_system(st, proc=fake.root, now=100.0)
    advance(fake)                                     # primero los deltas nuevos…
    (fake.root / "42" / "fdinfo" / "0").chmod(0o000)  # …ahora el kernel «oculta» el fd
    try:
        s1 = sample_system(st, proc=fake.root, now=101.0)
    finally:
        (fake.root / "42" / "fdinfo" / "0").chmod(0o644)
    p42 = next(p for p in s1.procs if p.pid == 42)
    assert p42.gpu_pct is None                        # degradación elegante §4


# ------------------------------------------------------------- matar proceso


def test_kill_process_guards_and_signal(monkeypatch, tmp_path):
    import signal

    llamadas = []
    monkeypatch.setattr(triop.os, "kill", lambda pid, sig: llamadas.append((pid, sig)))

    ok, msg = triop.kill_process(4242, killer=triop.os.kill, own_pid=999)
    assert ok and llamadas == [(4242, signal.SIGTERM)]

    ok, _ = triop.kill_process(1, killer=triop.os.kill, own_pid=999)
    assert not ok                                   # nunca init
    ok, _ = triop.kill_process(999, killer=triop.os.kill, own_pid=999)
    assert not ok                                   # nunca a nosotros mismos

    def permiso_denegado(pid, sig):
        raise PermissionError(1, "Operación no permitida")

    ok, msg = triop.kill_process(500, killer=permiso_denegado, own_pid=1)
    assert not ok and "permiso" in msg.lower()

    def inexistente(pid, sig):
        raise ProcessLookupError()

    ok, msg = triop.kill_process(500, killer=inexistente, own_pid=1)
    assert not ok


def test_ctrl_k_binding_registered():
    pares = [(b.key if hasattr(b, "key") else b[0],
              b.action if hasattr(b, "action") else b[1])
             for b in triop.TriopApp.BINDINGS]
    claves = {k for k, _ in pares}
    assert "ctrl+k" in claves
    assert any("kill" in str(a) for _, a in pares)


# --------------------------------------------------------------- TUI (pilot)


def test_summary_block_vertical_layout():
    from triop import DEFAULT_PALETTE, DASH, ProcRow, Snapshot, Totals, summary_block

    snap = Snapshot(
        Totals(cpu_pct=50.0, ram_used=4 * 2**30, ram_total=8 * 2**30,
               gpu_busy_pct=None, gpu_mem_used=512 * 2**20),
        procs=[ProcRow(pid=1, user="u", name="n", cpu_pct=1.0, rss=1,
                       gpu_pct=None, vram=None)],
    )
    txt = summary_block(snap, DEFAULT_PALETTE, interval=0.5, sort_key="gpu",
                        sort_desc=True, now_str="10:42:03", bar_width=10)
    lines = [l for l in txt.plain.splitlines() if l.strip()]
    assert len(lines) == 5                                   # 4 métricas + estado
    heads = [l.strip().split()[0] for l in lines[:4]]
    assert heads == ["CPU", "RAM", "GPU", "VRAM"]            # cada ítem debajo del otro
    assert lines[0].count("█") == 5                          # barra 50% de ancho 10
    assert DASH in lines[2]                                  # GPU sin datos -> –
    assert "512" in lines[3]                                 # VRAM en GiB->MiB
    assert "10:42:03" in lines[4] and "GPU" in lines[4]
    assert "desc" in lines[4] and "0.50s" in lines[4]


class TestPilot(unittest.IsolatedAsyncioTestCase):
    async def test_app_mounts_three_snapshots_rows_and_sort_keys(self):
        from textual.widgets import DataTable

        # interval grande: el timer queda inerte y los refrescos son solo los
        # explícitos -> determinista aunque el sistema esté cargado
        app = triop.TriopApp(interval=60)
        async with app.run_test(size=(110, 32)) as pilot:
            await pilot.pause()
            app.refresh_data()
            await pilot.pause()
            app.refresh_data()
            await pilot.pause()

            assert app.sample_count >= 3                    # DoD §8.3
            table = app.query_one(DataTable)
            assert table.row_count >= 1                     # filas reales del sistema

            await pilot.press("3")                          # ordenar por GPU
            assert app.sort_key == "gpu"
            assert app.sort_desc is True
            await pilot.press("r")                          # invertir
            assert app.sort_desc is False
            await pilot.press("r")

            await pilot.press("1")
            assert app.sort_key == "cpu"
            await pilot.press("2")
            assert app.sort_key == "mem"
            await pilot.press("4")
            assert app.sort_key == "pid"
            await pilot.press("5")
            assert app.sort_key == "name"

            # clamp al mínimo partiendo cerca del borde (independiente del
            # interval inicial del test) y un paso arriba coherente
            app.interval = triop.MIN_INTERVAL + triop.INTERVAL_STEP
            await pilot.press("-")
            assert app.interval == pytest.approx(triop.MIN_INTERVAL)
            await pilot.press("+")
            assert app.interval == pytest.approx(triop.MIN_INTERVAL + triop.INTERVAL_STEP)

            # preservación del cursor tras refresco (§5.4); tolera que el pid
            # desaparezca del sistema entre muestras (churn real de procesos)
            if table.row_count >= 2:
                table.move_cursor(row=1)
                sel_pid = str(table.get_row_at(1)[0])
                app.refresh_data()
                await pilot.pause()
                cur_pid = str(table.get_row_at(table.cursor_row)[0])
                procs_now = app.last_snapshot.procs if app.last_snapshot else []
                still_present = any(p.pid == int(sel_pid) for p in procs_now)
                assert cur_pid == sel_pid or not still_present
