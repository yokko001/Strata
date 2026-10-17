#!/usr/bin/env python3
"""Emit a path-sanitized Linux host inventory for MI50 benchmark reports.

Example:
    python3 tools/mi50/host_inventory.py \
        --path pack=/srv/strata/packs/q2_0 \
        --path model=/srv/models/model.gguf > host-inventory.json

Only the caller-supplied label is emitted for a path.  Mount points, device
names, host names and the original path are deliberately omitted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _kib_to_gib(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"\s*(\d+)\s+kB\s*$", value, re.IGNORECASE)
    return round(int(match.group(1)) / 2**20, 2) if match else None


def parse_meminfo(text: str) -> dict[str, float | None]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = value.strip()
    swap_total = _kib_to_gib(values.get("SwapTotal"))
    swap_free = _kib_to_gib(values.get("SwapFree"))
    return {
        "total_gib": _kib_to_gib(values.get("MemTotal")),
        "available_gib": _kib_to_gib(values.get("MemAvailable")),
        "swap_total_gib": swap_total,
        "swap_used_gib": round(swap_total - swap_free, 2)
        if swap_total is not None and swap_free is not None
        else None,
        "hugepages_total": int(values["HugePages_Total"].split()[0])
        if values.get("HugePages_Total")
        else None,
    }


def selected_kernel_setting(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\[([^]]+)\]", text)
    return match.group(1) if match else text.split()[0]


def cpu_inventory(sys_root: Path = Path("/sys")) -> dict[str, Any]:
    try:
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = set(range(os.cpu_count() or 0))
    cores: set[tuple[str, str]] = set()
    sockets: set[str] = set()
    for cpu in sorted(allowed):
        topology = sys_root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
        core = _read(topology / "core_id")
        package = _read(topology / "physical_package_id")
        if core is not None and package is not None:
            cores.add((package, core))
            sockets.add(package)
    return {
        "allowed_logical_cpus": len(allowed),
        "physical_cores_visible": len(cores) or None,
        "sockets_visible": len(sockets) or None,
    }


def _run_json(command: list[str]) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def classify_mount(fstype: str | None, source: str | None) -> dict[str, Any]:
    fs = (fstype or "unknown").lower()
    network_types = {"9p", "cifs", "fuse.sshfs", "nfs", "nfs4", "smb3", "virtiofs"}
    return {
        "filesystem": fs,
        "network_filesystem": fs in network_types,
        "block_backed": bool(source and source.startswith("/dev/")),
    }


def _flatten_blockdevices(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        out.append(node)
        out.extend(_flatten_blockdevices(node.get("children") or []))
    return out


def path_inventory(path: Path) -> dict[str, Any]:
    """Return storage properties without returning the path, mount point or device."""
    result: dict[str, Any] = {"exists": path.exists()}
    if not result["exists"]:
        return result
    mount = _run_json(["findmnt", "--json", "--target", str(path), "--output", "FSTYPE,SOURCE"])
    filesystems = (mount or {}).get("filesystems") or []
    if not filesystems:
        return result
    entry = filesystems[0]
    source = entry.get("source")
    result.update(classify_mount(entry.get("fstype"), source))
    if not result["block_backed"]:
        return result
    blocks = _run_json(["lsblk", "--json", "--bytes", "--output", "NAME,KNAME,PKNAME,TYPE,ROTA,RA,SCHED,SIZE"])
    nodes = _flatten_blockdevices((blocks or {}).get("blockdevices") or [])
    wanted = Path(source).name if source else ""
    node = next((n for n in nodes if n.get("name") == wanted or n.get("kname") == wanted), None)
    if node and node.get("pkname"):
        node = next((n for n in nodes if n.get("name") == node.get("pkname")), node)
    if node:
        result.update(
            {
                "rotational": bool(node.get("rota")) if node.get("rota") is not None else None,
                "read_ahead_kib": node.get("ra"),
                "scheduler": node.get("sched"),
                "size_bytes": node.get("size"),
            }
        )
    return result


def inventory(paths: list[tuple[str, Path]]) -> dict[str, Any]:
    mem_text = _read(Path("/proc/meminfo")) or ""
    return {
        "schema": 1,
        "cpu": cpu_inventory(),
        "memory": parse_meminfo(mem_text),
        "vm": {
            "swappiness": int(value) if (value := _read(Path("/proc/sys/vm/swappiness"))) else None,
            "transparent_hugepage": selected_kernel_setting(
                _read(Path("/sys/kernel/mm/transparent_hugepage/enabled"))
            ),
            "transparent_hugepage_defrag": selected_kernel_setting(
                _read(Path("/sys/kernel/mm/transparent_hugepage/defrag"))
            ),
        },
        "storage": {label: path_inventory(path) for label, path in paths},
    }


def parse_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or not raw_path:
        raise argparse.ArgumentTypeError("label must be non-empty and contain only A-Z, a-z, 0-9, _, . or -")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", type=parse_path, default=[], metavar="LABEL=PATH")
    args = parser.parse_args()
    print(json.dumps(inventory(args.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
