import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("mi50") / "host_inventory.py"
SPEC = importlib.util.spec_from_file_location("mi50_host_inventory", MODULE_PATH)
assert SPEC and SPEC.loader
host_inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host_inventory)


class HostInventoryTest(unittest.TestCase):
    def test_parse_meminfo(self):
        parsed = host_inventory.parse_meminfo(
            "MemTotal:       22020096 kB\n"
            "MemAvailable:   10485760 kB\n"
            "SwapTotal:       4000000 kB\n"
            "SwapFree:        1902848 kB\n"
            "HugePages_Total:       0\n"
        )
        self.assertEqual(parsed["total_gib"], 21.0)
        self.assertEqual(parsed["available_gib"], 10.0)
        self.assertEqual(parsed["swap_used_gib"], 2.0)
        self.assertEqual(parsed["hugepages_total"], 0)

    def test_selected_kernel_setting(self):
        self.assertEqual(host_inventory.selected_kernel_setting("always [madvise] never"), "madvise")
        self.assertEqual(host_inventory.selected_kernel_setting("none mq-deadline [kyber]"), "kyber")

    def test_mount_classification(self):
        self.assertEqual(
            host_inventory.classify_mount("cifs", "//server/share"),
            {"filesystem": "cifs", "network_filesystem": True, "block_backed": False},
        )
        self.assertEqual(
            host_inventory.classify_mount("ext4", "/dev/sdb1"),
            {"filesystem": "ext4", "network_filesystem": False, "block_backed": True},
        )

    def test_cpu_inventory_uses_package_and_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cpu, package, core in ((0, 0, 0), (1, 0, 1)):
                topology = root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text(str(package))
                (topology / "core_id").write_text(str(core))
            with mock.patch.object(host_inventory.os, "sched_getaffinity", lambda _pid: {0, 1}, create=True):
                self.assertEqual(
                    host_inventory.cpu_inventory(root),
                    {"allowed_logical_cpus": 2, "physical_cores_visible": 2, "sockets_visible": 1},
                )

    def test_missing_path_does_not_echo_it(self):
        secret_path = Path("/private/example/model.gguf")
        self.assertEqual(host_inventory.path_inventory(secret_path), {"exists": False})


if __name__ == "__main__":
    unittest.main()
