#!/usr/bin/env python3
"""triop — monitor TUI de CPU / RAM / GPU por proceso.

Implementación del MVP descrito en triop-plan.md (§4 alcance, §5 diseño,
§8 Definition of Done). Single-file por decisión de diseño (§5.2).

Fuentes de datos (todo /proc, sin dependencias externas de sistema):
  - CPU total:      /proc/stat (delta idle vs total)
  - CPU por proceso:/proc/<pid>/stat campos utime+stime (ticks / CLK_TCK)
  - RAM:            /proc/meminfo (MemTotal-MemAvailable) y statm resident×pagesize
  - GPU por proceso:/proc/<pid>/fdinfo/<fd> claves drm-engine-* y drm-total-*
                    (mismo mecanismo que nvtop para i915/amdgpu/xe)

Uso:  triop [--interval S] [--sort KEY] [--print [N]] [--version]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from pwd import getpwuid

from textual.app import App
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Static

VERSION = "0.1.0"

PROC = Path("/proc")
MIN_INTERVAL = 0.25          # plan §5.5: mínimo 250 ms
DEFAULT_INTERVAL = 1.0       # plan §5.4: refresco 1 s por defecto
INTERVAL_STEP = 0.25
GPU_RECHECK_EVERY = 10       # plan §5.3: revisar procesos GPU-idle cada 10 ciclos
DASH = "–"                   # marcador «sin datos» (plan §4)

SORT_COLUMNS = ("cpu", "mem", "gpu", "pid", "name")

CLK_TCK = float(os.sysconf("SC_CLK_TCK"))

_ENGINE_RE = re.compile(r"^drm-engine-(\S+):\s+(\d+) ns", re.M)
_TOTAL_RE = re.compile(r"^drm-total-(\S+):\s+(\d+) KiB", re.M)


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------


@dataclass
class Totals:
    """Totales globales de la cabecera (plan §4)."""

    cpu_pct: float | None = None
    ram_used: int = 0
    ram_total: int = 0
    gpu_busy_pct: float | None = None
    gpu_mem_used: int | None = None


@dataclass
class ProcRow:
    """Una fila de la tabla de procesos."""

    pid: int
    user: str
    name: str
    cpu_pct: float | None
    rss: int | None                 # bytes residentes
    gpu_pct: float | None           # None => '–'
    vram: int | None                # bytes; None => '–'
    engines_delta_ns: dict[str, int] = field(default_factory=dict)


@dataclass
class Snapshot:
    totals: Totals
    procs: list[ProcRow]


@dataclass
class GpuInfo:
    """Lectura acumulada del fdinfo de un proceso (ns absolutos, KiB)."""

    engines_ns: dict[str, int]      # suma de todos los fd del proceso
    vram_kib: int                   # mejor clave drm-total-* (sin stolen)


@dataclass
class SamplerState:
    """Estado entre muestras: valores previos para calcular deltas."""

    t: float | None = None
    cpu_total: tuple[int, int] | None = None      # (idle_all, total_all)
    ticks: dict[int, int] = field(default_factory=dict)
    gpu: dict[int, dict[str, int]] = field(default_factory=dict)   # incluye {} = ya comprobado sin DRM
    vram: dict[int, int] = field(default_factory=dict)             # KiB
    hot: set[int] = field(default_factory=set)                     # pids con actividad GPU reciente
    cycle: int = 0


# --------------------------------------------------------------------------
# Lectores /proc (funciones puras sobre rutas o texto: testeables)
# --------------------------------------------------------------------------


def read_cpu_total(proc: Path = PROC) -> tuple[int, int]:
    """Primera línea de /proc/stat → (idle_all, total_all) en jiffies."""
    first = (proc / "stat").read_text().splitlines()[0]
    nums = [int(x) for x in first.split()[1:] if x.isdigit()]
    if len(nums) < 4:
        raise ValueError(f"/proc/stat malformado: {first!r}")
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)   # idle + iowait
    return idle, sum(nums)


def cpu_busy_pct(prev: tuple[int, int], cur: tuple[int, int]) -> float | None:
    """Plan §5.3a: 100 × (1 − Δidle/Δtotal), acotado a [0, 100]."""
    di = cur[0] - prev[0]
    dt = cur[1] - prev[1]
    if dt <= 0 or di < 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - di / dt)))


def read_meminfo(proc: Path = PROC) -> tuple[int, int]:
    """Plan §5.3c: (MemTotal − MemAvailable, MemTotal) en bytes."""
    vals: dict[str, int] = {}
    for line in (proc / "meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                vals[parts[0][:-1]] = int(parts[1]) * 1024
            except ValueError:
                continue
    total = vals.get("MemTotal", 0)
    avail = vals.get("MemAvailable", vals.get("MemFree", 0))
    return max(0, total - avail), total


def _split_stat(text: str) -> tuple[str, list[str]]:
    """Divide /proc/<pid>/stat respetando paréntesis dentro de comm.

    Devuelve (comm, campos_desde_field_3).
    """
    head, _, tail = text.partition("(")
    comm, _, rest = tail.rpartition(")")
    return comm.strip(), rest.split()


def ticks_from_stat(text: str) -> int:
    fields = _split_stat(text)[1]
    if len(fields) < 13:
        raise ValueError("stat demasiado corto")
    return int(fields[11]) + int(fields[12])             # utime (f14) + stime (f15)


def comm_from_stat(text: str) -> str:
    return _split_stat(text)[0]


def cmdline_name(pid_dir: Path) -> str:
    """Fallback de nombre (plan §5.3g): cmdline truncada."""
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:64]


def _pick_vram(totals_kib: dict[str, int]) -> int:
    """Plan §5.3e: preferir drm-total-system0; si falta, la primera clave
    drm-total-* que no sea *stolen*; 0 si no hay nada utilizable."""
    if "system0" in totals_kib:
        return totals_kib["system0"]
    for key, val in totals_kib.items():
        if "stolen" not in key:
            return val
    return 0


def read_gpu(pid_dir: Path) -> GpuInfo | None:
    """Lee todos los fdinfo del proceso buscando claves DRM (plan §5.3d/e).

    Devuelve None si no hay fdinfo legible o ninguna línea drm-*.
    """
    fd_dir = pid_dir / "fdinfo"
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return None
    engines: dict[str, int] = defaultdict(int)
    best_vram = 0
    found = False
    for fd in fds[:512]:                                 # cota defensiva
        try:
            txt = fd.read_text(errors="replace")
        except OSError:
            continue                                     # fd oculto por el kernel
        if "drm-" not in txt:
            continue
        found = True
        for m in _ENGINE_RE.finditer(txt):
            engines[m.group(1)] += int(m.group(2))
        best_vram = max(best_vram, _pick_vram(
            {m.group(1): int(m.group(2)) for m in _TOTAL_RE.finditer(txt)}
        ))
    if not found:
        return None
    return GpuInfo(dict(engines), best_vram)


def user_from_uid(uid: int) -> str:
    try:
        return getpwuid(uid).pw_name
    except (KeyError, TypeError, ValueError, OverflowError):
        return str(uid)


def fmt_mem(size: int | None) -> str:
    if size is None:
        return DASH
    for unit, factor in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if size >= factor:
            return f"{size / factor:.1f} {unit}"
    return f"{size} B"


def fmt_pct(value: float | None) -> str:
    return DASH if value is None else f"{value:.1f}"


# --------------------------------------------------------------------------
# Paleta pywal + bloque de totales vertical (restyling §5.6)
# --------------------------------------------------------------------------

WAL_JSON = Path("~/.cache/wal/colors.json").expanduser()


@dataclass(frozen=True)
class Palette:
    bg: str
    fg: str
    dim: str
    track: str
    cpu: str
    ram: str
    gpu: str
    vram: str


def _blend(hex_a: str, hex_b: str, t: float) -> str:
    """Mezcla lineal #rrggbb; t=0 devuelve a, t=1 devuelve b."""
    try:
        a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    except (ValueError, IndexError):
        return hex_a
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


DEFAULT_PALETTE = Palette(
    bg="#11131a", fg="#d8dce6", dim="#5c6274", track="#232735",
    cpu="#61b3ff", ram="#79d68a", gpu="#c39bff", vram="#e2b465",
)


def load_palette(path: Path = WAL_JSON) -> Palette:
    """Lee pywal (formato anidado pywal16 o plano legacy); fallback neutro."""
    fallback = DEFAULT_PALETTE
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback
    colors = data.get("colors", data)
    special = data.get("special", {})
    bg = special.get("background") or data.get("background") or fallback.bg
    fg = special.get("foreground") or data.get("foreground") or fallback.fg
    pick = lambda idx: colors.get(f"color{idx}")   # noqa: E731
    return Palette(
        bg=bg, fg=fg,
        dim=pick(8) or _blend(bg, fg, 0.45),
        track=_blend(bg, fg, 0.16),
        cpu=pick(6) or fallback.cpu,
        ram=pick(5) or fallback.ram,
        gpu=pick(4) or fallback.gpu,
        vram=pick(3) or fallback.vram,
    )


def _metric_bar(pct: float | None, width: int, color: str, track: str):
    from rich.text import Text

    if pct is None:
        return Text("")
    filled = int(round(width * max(0.0, min(100.0, pct)) / 100))
    bar = Text("█" * filled, style=color)
    bar.append("░" * (width - filled), style=track)
    return bar


def _metric_line(label: str, color: str, fg: str, value: str,
                 pct: float | None, pal: Palette, bar_width: int):
    from rich.text import Text

    strong = pct is not None and pct >= 85
    line = Text(f"{label:<5}", style=f"bold {color}")
    bar = _metric_bar(pct, bar_width, color, pal.track)
    if bar.plain:
        line.append_text(bar)
        line.append(" ")
    else:
        line.append(" " * (bar_width + 1))
    line.append(value, style=f"bold {color}" if strong else fg)
    return line


def summary_block(snapshot: Snapshot, pal: Palette, *, interval: float,
                  sort_key: str, sort_desc: bool, now_str: str = "",
                  bar_width: int = 22):
    """Totales apilados en vertical (una métrica por línea, estilo htop)."""
    from rich.text import Text

    t = snapshot.totals
    ram_pct = (100.0 * t.ram_used / t.ram_total) if t.ram_total else None
    metrics = [
        ("CPU", pal.cpu, t.cpu_pct,
         DASH if t.cpu_pct is None else f"{t.cpu_pct:.1f}%"),
        ("RAM", pal.ram, ram_pct,
         f"{fmt_mem(t.ram_used)} / {fmt_mem(t.ram_total)}"
         + (f" · {ram_pct:.0f}%" if ram_pct is not None else "")),
        ("GPU", pal.gpu, t.gpu_busy_pct,
         DASH if t.gpu_busy_pct is None else f"{t.gpu_busy_pct:.1f}%"),
        ("VRAM", pal.vram, None,
         DASH if t.gpu_mem_used is None else fmt_mem(t.gpu_mem_used)),
    ]
    out = Text()
    for i, (label, color, pct, value) in enumerate(metrics):
        if i:
            out.append("\n")
        out.append_text(_metric_line(label, color, pal.fg, value, pct, pal, bar_width))
    orden = f"{sort_key.upper()} {'desc' if sort_desc else 'asc'}"
    status = Text(f"{now_str} · {interval:.2f}s ▮ orden {orden}".strip(), style=pal.dim)
    out.append("\n")
    out.append_text(status)
    return out


# --------------------------------------------------------------------------
# Muestreador (plan §5.3 y §5.4)
# --------------------------------------------------------------------------


def sample_system(
    state: SamplerState,
    proc: Path = PROC,
    now: float | None = None,
    pagesize: int | None = None,
) -> Snapshot:
    """Una pasada por /proc; devuelve Snapshot y actualiza `state`.

    `now` y `pagesize` son inyectables para tests headless deterministas.
    """
    if now is None:
        now = time.monotonic()
    if pagesize is None:
        pagesize = os.sysconf("SC_PAGESIZE")

    dt = None if state.t is None else now - state.t
    cpu_now = read_cpu_total(proc)
    try:
        ram_used, ram_total = read_meminfo(proc)
    except OSError:
        ram_used, ram_total = 0, 0

    full_scan = state.cycle % GPU_RECHECK_EVERY == 0
    new_ticks: dict[int, int] = {}
    new_gpu: dict[int, dict[str, int]] = {}
    new_vram: dict[int, int] = {}
    new_hot: set[int] = set()
    total_engine_delta_ns = 0
    any_vram_seen = False
    rows: list[ProcRow] = []

    try:
        pids = sorted(int(p.name) for p in proc.iterdir() if p.name.isdigit())
    except OSError:
        pids = []

    for pid in pids:
        d = proc / str(pid)
        try:
            stat_txt = (d / "stat").read_text()
            ticks = ticks_from_stat(stat_txt)
            name = comm_from_stat(stat_txt)
            uid = d.stat().st_uid            # plan §5.3g corregido: stat no tiene uid
            rss = int((d / "statm").read_text().split()[1]) * pagesize
        except (OSError, ValueError, IndexError):
            continue                                     # murió muestreando: saltar

        if not name:
            name = cmdline_name(d) or f"[{pid}]"

        # --- GPU (con optimización §5.3: solo re-delta de pids 'hot') ---
        need_read = full_scan or pid in state.hot or pid not in state.gpu
        gi = read_gpu(d) if need_read else None
        vram_kib: int | None = None
        engines_delta: dict[str, int] = {}
        gpu_pct: float | None = None
        pid_engine_delta = 0

        if gi is not None:
            new_gpu[pid] = dict(gi.engines_ns)
            vram_kib = gi.vram_kib
            new_vram[pid] = vram_kib
            if vram_kib > 0:
                any_vram_seen = True
            prev_eng = state.gpu.get(pid)
            if prev_eng is not None:
                for eng in set(gi.engines_ns) | set(prev_eng):
                    delta = gi.engines_ns.get(eng, 0) - prev_eng.get(eng, 0)
                    if delta > 0:
                        engines_delta[eng] = delta
                        pid_engine_delta += delta
            if pid_engine_delta > 0 or (prev_eng is None and gi.engines_ns):
                new_hot.add(pid)   # hot tentativo: un recién visto debe confirmarse midiendo el ciclo siguiente
                total_engine_delta_ns += max(0, pid_engine_delta)
        elif pid in state.gpu and state.gpu[pid]:
            # Proceso frío fuera de su turno de relectura: arrastrar estado.
            new_gpu[pid] = state.gpu[pid]
            if pid in state.vram:
                vram_kib = state.vram[pid]
                new_vram[pid] = vram_kib
        else:
            # Comprobado y sin DRM (o fdinfo ilegible): marcar como revisado.
            new_gpu[pid] = {}
            if pid in state.vram:
                vram_kib = state.vram[pid]
                new_vram[pid] = vram_kib

        if dt and dt > 0 and pid_engine_delta > 0:
            gpu_pct = min(100.0, 100.0 * pid_engine_delta / (dt * 1e9))

        # --- CPU% por proceso (plan §5.3b) ---
        prev_ticks = state.ticks.get(pid)
        cpu_pct: float | None = None
        if prev_ticks is not None and dt and dt > 0:
            cpu_pct = max(0.0, 100.0 * ((ticks - prev_ticks) / CLK_TCK) / dt)
        new_ticks[pid] = ticks

        rows.append(ProcRow(
            pid=pid,
            user=user_from_uid(uid),
            name=name,
            cpu_pct=cpu_pct,
            rss=rss,
            gpu_pct=gpu_pct,
            vram=vram_kib * 1024 if vram_kib is not None else None,
            engines_delta_ns=engines_delta,
        ))

    # --- Totales (plan §5.3a/f) ---
    cpu_tot = cpu_busy_pct(state.cpu_total, cpu_now) if state.cpu_total else None
    if dt and dt > 0 and total_engine_delta_ns > 0:
        gpu_busy: float | None = min(100.0, 100.0 * total_engine_delta_ns / (dt * 1e9))
    elif any_vram_seen or new_vram:
        gpu_busy = 0.0                                   # GPU presente pero idle
    else:
        gpu_busy = None                                  # sin datos GPU en absoluto
    gpu_mem = sum(new_vram.values()) * 1024 if new_vram else None

    # --- Commit del estado ---
    state.t = now
    state.cpu_total = cpu_now
    state.ticks = new_ticks
    state.gpu = new_gpu
    state.vram = new_vram
    state.hot = new_hot
    state.cycle += 1

    return Snapshot(Totals(cpu_pct=cpu_tot, ram_used=ram_used, ram_total=ram_total,
                           gpu_busy_pct=gpu_busy, gpu_mem_used=gpu_mem), rows)


# --------------------------------------------------------------------------
# Ordenación y presentación
# --------------------------------------------------------------------------


def _sort_val(row: ProcRow, key: str):
    if key == "pid":
        return row.pid
    if key == "name":
        return row.name.casefold()
    value = {"cpu": row.cpu_pct, "mem": row.rss, "gpu": row.gpu_pct}[key]
    return float("-inf") if value is None else value     # '–' siempre al fondo en desc


def sort_rows(rows: list[ProcRow], key: str, desc: bool) -> list[ProcRow]:
    return sorted(rows, key=lambda r: _sort_val(r, key), reverse=desc)


def kill_process(
    pid: int,
    *,
    sig: int | None = None,
    killer=None,
    own_pid: int | None = None,
) -> tuple[bool, str]:
    """Señal al proceso (SIGTERM por defecto); nunca a init ni a triop."""
    send = killer if killer is not None else os.kill
    own = os.getpid() if own_pid is None else own_pid
    if sig is None:
        sig = signal.SIGTERM
    nombre = signal.Signals(sig).name
    if pid <= 1:
        return False, f"pid {pid} intocable"
    if pid == own:
        return False, "triop no se suicida"
    try:
        send(pid, sig)
    except PermissionError:
        return False, f"sin permiso para matar {pid} (prueba sudo)"
    except ProcessLookupError:
        return False, f"{pid} ya no existe"
    return True, f"{nombre} enviado a {pid}"


def summary_line(snap: Snapshot) -> str:
    t = snap.totals
    ram_pct = f" ({t.ram_used * 100 // t.ram_total}%)" if t.ram_total else ""
    return (
        f"triop {VERSION} — CPU {fmt_pct(t.cpu_pct)}%  "
        f"RAM {fmt_mem(t.ram_used)}/{fmt_mem(t.ram_total)}{ram_pct}  "
        f"GPU {fmt_pct(t.gpu_busy_pct)}%  VRAM {fmt_mem(t.gpu_mem_used)}"
    )


def run_print(n: int, interval: float, sort_key: str, proc: Path = PROC) -> int:
    """Modo no interactivo (plan §4): dos muestras separadas por `interval`,
    imprime una tabla única y sale."""
    state = SamplerState()
    sample_system(state, proc=proc)
    time.sleep(interval)
    snap = sample_system(state, proc=proc)

    headers = ("PID", "USUARIO", "NOMBRE", "CPU%", "MEM", "GPU%", "VRAM")
    aligns = (">", "<", "<", ">", ">", ">", ">")
    cells: list[tuple[str, ...]] = [
        (str(r.pid), r.user, r.name, fmt_pct(r.cpu_pct),
         fmt_mem(r.rss), fmt_pct(r.gpu_pct), fmt_mem(r.vram))
        for r in sort_rows(snap.procs, sort_key, True)[:n]
    ]
    widths = [
        max(len(headers[i]), *(len(c[i]) for c in cells)) if cells else len(headers[i])
        for i in range(len(headers))
    ]
    line = "-+-".join("-" * w for w in widths)

    out = [f"{summary_line(snap)}  ·  {time.strftime('%H:%M:%S')}"]
    out.append(" | ".join(f"{h:{a}{w}}" for h, a, w in zip(headers, aligns, widths)))
    out.append(line.replace("|", "+"))
    out += [" | ".join(f"{c:{a}{w}}" for c, a, w in zip(row, aligns, widths)) for row in cells]
    print("\n".join(out))
    return 0


# --------------------------------------------------------------------------
# TUI (Textual) — plan §5.4/§5.5
# --------------------------------------------------------------------------

SUMMARY_COLUMNS = ("PID", "USUARIO", "NOMBRE", "CPU%", "MEM", "GPU%", "VRAM")


class TriopApp(App):
    """App principal: cabecera de totales + tabla ordenable + pie."""

    TITLE = f"triop {VERSION}"
    CSS = """
    App { background: ansi_default; }
    Screen { background: ansi_default; padding: 1 2; }
    #summary { dock: top; height: auto; padding: 0 0 1 0; background: ansi_default; }
    #procs { height: 1fr; border: none;
            scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    DataTable { background: ansi_default; }
    Footer { background: ansi_default; }
    #procs .scrollbar,
    #procs .scrollbar--vertical,
    #procs .scrollbar--horizontal { background: ansi_default; }
    """
    BINDINGS = [
        ("q", "quit", "Salir"),
        ("ctrl+k", "kill_selected", "Matar"),
        ("ctrl+shift+k", "force_kill_selected", "Forzar"),
        ("1", "sort('cpu')", "CPU"),
        ("2", "sort('mem')", "MEM"),
        ("3", "sort('gpu')", "GPU"),
        ("4", "sort('pid')", "PID"),
        ("5", "sort('name')", "Nombre"),
        ("r", "reverse", "Invertir"),
        ("-", "interval_down", "Intervalo−"),
        ("+", "interval_up", "Intervalo+"),
        ("j", "cursor_down", ""),
        ("k", "cursor_up", ""),
    ]

    def __init__(self, interval: float = DEFAULT_INTERVAL, sort_key: str = "cpu",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.interval = max(MIN_INTERVAL, float(interval))
        self.sort_key = sort_key if sort_key in SORT_COLUMNS else "cpu"
        self.sort_desc = True
        self.state = SamplerState()
        self.sample_count = 0
        self.last_snapshot: Snapshot | None = None
        self._timer = None
        self.palette = load_palette()
        try:
            self.register_theme(Theme(
                name="triop-wal",
                primary=self.palette.cpu,
                secondary=self.palette.vram,
                accent=self.palette.gpu,
                success=self.palette.ram,
                warning=self.palette.fg,
                error=self.palette.fg,
                foreground=self.palette.fg,
                background=self.palette.bg,
                surface=self.palette.bg,
                panel=self.palette.bg,
                boost=self.palette.track,
                dark=True,
                variables={
                    "footer-background": "ansi_default",
                    "footer-key-background": "ansi_default",
                    "footer-description-background": "ansi_default",
                },
            ))
            self.theme = "triop-wal"
        except Exception:
            pass                       # textual sin API de temas: quedan defaults
        # Passthrough estilo htop: sin este swap, ANSIToTruecolor convierte los
        # fondos «default» al gris del tema y se pierde el fondo de la terminal.
        try:
            from textual.filter import LineFilter

            class _NoOpFilter(LineFilter):
                def apply(self, segments, background):
                    return segments

            for i, f in enumerate(self._filters):
                if isinstance(f, LineFilter):
                    self._filters[i] = _NoOpFilter()
        except Exception:
            pass

    def compose(self):
        yield Static("", id="summary")
        yield DataTable(id="procs", cursor_type="row")
        yield Footer()

    def action_kill_selected(self) -> None:
        self._kill_selected(signal.SIGTERM)

    def action_force_kill_selected(self) -> None:
        self._kill_selected(signal.SIGKILL)

    def _kill_selected(self, sig: int) -> None:
        table = self.query_one(DataTable)
        try:
            if not table.row_count or not 0 <= table.cursor_row < table.row_count:
                self.notify("no hay proceso seleccionado", severity="warning")
                return
            pid = int(str(table.get_row_at(table.cursor_row)[0]))
        except Exception:
            return
        ok, msg = kill_process(pid, sig=sig)
        self.notify(("✓ " if ok else "✗ ") + msg,
                    severity="information" if ok else "error")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(*SUMMARY_COLUMNS)
        self._timer = self.set_interval(self.interval, self.refresh_data)
        self.refresh_data()

    # -- núcleo ------------------------------------------------------------

    def refresh_data(self) -> None:
        snap = sample_system(self.state)
        self.sample_count += 1
        self.last_snapshot = snap
        self._update_summary()
        self._fill_table(snap)

    def _update_summary(self) -> None:
        if self.last_snapshot is None:
            return
        self.query_one("#summary", Static).update(summary_block(
            self.last_snapshot, self.palette,
            interval=self.interval, sort_key=self.sort_key,
            sort_desc=self.sort_desc, now_str=time.strftime("%H:%M:%S")))

    def _fill_table(self, snap: Snapshot) -> None:
        from rich.text import Text

        table = self.query_one(DataTable)
        selected: str | None = None
        try:
            if 0 <= table.cursor_row < table.row_count:
                cell = table.get_row_at(table.cursor_row)[0]
                selected = str(cell)
        except Exception:
            selected = None

        table.clear()
        for r in sort_rows(snap.procs, self.sort_key, self.sort_desc):
            table.add_row(
                Text(str(r.pid)), Text(r.user), Text(r.name),
                Text(fmt_pct(r.cpu_pct)), Text(fmt_mem(r.rss)),
                Text(fmt_pct(r.gpu_pct)), Text(fmt_mem(r.vram)),
                key=str(r.pid),
            )
        # Restaurar cursor sobre el mismo PID tras repoblar (plan §5.4).
        if selected is not None:
            try:
                table.move_cursor(row=table.get_row_index(selected))
            except Exception:
                pass
        if table.row_count and table.cursor_row >= table.row_count:
            table.move_cursor(row=table.row_count - 1)

    # -- acciones §5.5 ------------------------------------------------------

    def action_sort(self, key: str) -> None:
        if self.sort_key == key:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_key = key
            self.sort_desc = True
        self._update_summary()
        if self.last_snapshot is not None:
            self._fill_table(self.last_snapshot)

    def action_reverse(self) -> None:
        self.action_sort(self.sort_key)

    def _change_interval(self, sign: int) -> None:
        self.interval = max(MIN_INTERVAL, round((self.interval + sign * INTERVAL_STEP) * 100) / 100)
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self.interval, self.refresh_data)
        self._update_summary()

    def action_interval_down(self) -> None:
        self._change_interval(-1)

    def action_interval_up(self) -> None:
        self._change_interval(+1)

    def action_cursor_down(self) -> None:
        self.query_one(DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(DataTable).action_cursor_up()


# --------------------------------------------------------------------------
# CLI (plan §5.4 main())
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triop",
        description="Monitor TUI de CPU / RAM / GPU por proceso.",
    )
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"segundos entre muestras (mínimo {MIN_INTERVAL})")
    parser.add_argument("--sort", choices=SORT_COLUMNS, default="cpu",
                        help="columna inicial de ordenación (por defecto CPU desc)")
    parser.add_argument("--print", dest="print_n", type=int, nargs="?", const=20,
                        default=None, metavar="N",
                        help="imprime una tabla de N filas y sale")
    parser.add_argument("--version", action="version", version=f"triop {VERSION}")
    args = parser.parse_args(argv)

    interval = args.interval
    if interval < MIN_INTERVAL:
        print(f"triop: intervalo {interval}s < mínimo {MIN_INTERVAL}s; usando {MIN_INTERVAL}s",
              file=sys.stderr)
        interval = MIN_INTERVAL

    if args.print_n is not None:
        if args.print_n < 1:
            parser.error("--print: N debe ser >= 1")
        try:
            return run_print(args.print_n, interval, args.sort)
        except KeyboardInterrupt:
            return 130

    try:
        TriopApp(interval=interval, sort_key=args.sort).run()
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
