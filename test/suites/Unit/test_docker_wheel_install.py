import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_UCM_WHEEL_DOCKERFILES = {
    "Dockerfile.ucm-mindie-ascend.a2-v2",
    "Dockerfile.ucm-sglang-ascend.a2-v0.5.18",
    "Dockerfile.ucm-sglang-cuda-v0.5.5",
    "Dockerfile.ucm-vllm-ascend.a2-latest",
    "Dockerfile.ucm-vllm-ascend.a2-v0.18.0",
    "Dockerfile.ucm-vllm-ascend.a2-v0.18.0glm5.1",
    "Dockerfile.ucm-vllm-ascend.a2-v0.20.2rc1",
    "Dockerfile.ucm-vllm-ascend.a2-v0.21.0rc1",
    "Dockerfile.ucm-vllm-ascend.a2-v0.22.1rc1",
    "Dockerfile.ucm-vllm-ascend.a2-v0.23.0",
    "Dockerfile.ucm-vllm-ascend.a2-v0.23.0rc1",
    "Dockerfile.ucm-vllm-ascend.a3-v0.18.0glm5.1",
    "Dockerfile.ucm-vllm-ascend.a3-v0.20.2rc1",
    "Dockerfile.ucm-vllm-ascend.a3-v0.21.0rc1",
    "Dockerfile.ucm-vllm-ascend.a3-v0.22.1rc1",
    "Dockerfile.ucm-vllm-ascend.a3-v0.23.0",
    "Dockerfile.ucm-vllm-ascend.a3-v0.23.0rc1",
    "Dockerfile.ucm-vllm-cuda-latest",
    "Dockerfile.ucm-vllm-cuda-v0.18.0",
    "Dockerfile.ucm-vllm-cuda-v0.20.2",
    "Dockerfile.ucm-vllm-cuda-v0.21.0",
}
INSTALL_COMMAND = "RUN pip install /workspace/package/uc_manager-*.whl"
POST_INSTALL_COMMAND = (
    "RUN if [ -f /workspace/package/install.sh ]; then \\\n"
    "        bash /workspace/package/install.sh{failure_policy}; \\\n"
    "    fi"
)
FATAL_POST_INSTALL_DOCKERFILES = {"Dockerfile.ucm-sglang-ascend.a2-v0.5.18"}


class DockerWheelInstallTest(unittest.TestCase):
    def _wheel_dockerfiles(self):
        dockerfiles = sorted((REPO_ROOT / "docker").glob("Dockerfile*"))
        return [
            path
            for path in dockerfiles
            if "uc_manager-*.whl" in path.read_text(encoding="utf-8")
        ]

    def _expected_engine_type(self, dockerfile):
        name = dockerfile.name
        if "mindie" in name:
            return "mindie"
        if "sglang" in name:
            return "sglang"
        if "vllm-ascend.a2" in name:
            return "vllm-ascend.a2"
        if "vllm-ascend.a3" in name:
            return "vllm-ascend.a3"
        if "vllm-cuda" in name:
            return "vllm-cuda"
        self.fail(f"unknown engine type for {dockerfile}")

    def test_ucm_wheel_dockerfiles_run_optional_package_install_hook(self):
        wheel_dockerfiles = self._wheel_dockerfiles()

        self.assertEqual(
            {path.name for path in wheel_dockerfiles}, EXPECTED_UCM_WHEEL_DOCKERFILES
        )
        for path in wheel_dockerfiles:
            text = path.read_text(encoding="utf-8")
            self.assertIn(INSTALL_COMMAND, text, path)
            # Older images deliberately tolerate optional hook failures;
            # the newer SGLang A2 image propagates them.
            failure_policy = (
                "" if path.name in FATAL_POST_INSTALL_DOCKERFILES else " || true"
            )
            self.assertIn(
                POST_INSTALL_COMMAND.format(failure_policy=failure_policy), text, path
            )
            self.assertNotIn("uc_manager-*.whl &&", text, path)
            self.assertNotIn("install_ucm_wheel.sh", text, path)

    def test_post_install_hooks_execute_the_checked_script(self):
        with tempfile.TemporaryDirectory(prefix="ucm-hook-") as directory:
            hook = Path(directory) / "install.sh"
            for path in self._wheel_dockerfiles():
                text = path.read_text(encoding="utf-8")
                start = text.index("RUN if [ -f /workspace/package/install.sh ]; then")
                end = text.index("\n    fi", start) + len("\n    fi")
                command = text[start + len("RUN ") : end].replace(
                    "/workspace/package/install.sh", shlex.quote(str(hook))
                )
                for exit_code in (None, 0, 7):
                    with self.subTest(dockerfile=path.name, hook_exit=exit_code):
                        hook.unlink(missing_ok=True)
                        if exit_code is not None:
                            hook.write_text(f"printf 'hook-ran\\n'\nexit {exit_code}\n")
                        result = subprocess.run(
                            ["bash", "-c", command],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(result.stderr, "")
                        self.assertEqual(
                            result.stdout, "" if exit_code is None else "hook-ran\n"
                        )
                        expected = (
                            exit_code or 0
                            if path.name in FATAL_POST_INSTALL_DOCKERFILES
                            else 0
                        )
                        self.assertEqual(result.returncode, expected)

    def test_ucm_wheel_dockerfiles_export_engine_type(self):
        wheel_dockerfiles = self._wheel_dockerfiles()

        self.assertEqual(
            {path.name for path in wheel_dockerfiles}, EXPECTED_UCM_WHEEL_DOCKERFILES
        )
        for path in wheel_dockerfiles:
            text = path.read_text(encoding="utf-8")
            expected = f"ENV UCM_ENGINE_TYPE={self._expected_engine_type(path)}"
            self.assertIn(expected, text, path)

    def test_ascend_a3_dockerfiles_build_a3_package(self):
        a3_dockerfiles = [
            REPO_ROOT / "docker" / "Dockerfile.ucm-vllm-ascend.a3-v0.18.0glm5.1",
            REPO_ROOT / "docker" / "Dockerfile.ucm-vllm-ascend.a3-v0.20.2rc1",
        ]

        for path in a3_dockerfiles:
            text = path.read_text(encoding="utf-8")
            self.assertIn("ENV UCM_ENGINE_TYPE=vllm-ascend.a3", text, path)
            self.assertIn(
                "bash /workspace/unified-cache-management/scripts/build_ascend.sh -p ascend-a3",
                text,
                path,
            )

    def test_no_vllm_017_dockerfiles_remain(self):
        self.assertEqual(
            sorted(path.name for path in (REPO_ROOT / "docker").glob("*v0.17.0*")),
            [],
        )
        workflow_text = (
            REPO_ROOT / ".github" / "workflows" / "pull-request.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Dockerfile.ucm-vllm-ascend.a2-v0.17.0", workflow_text)

    def test_no_deepseekv4_dockerfile_remains(self):
        self.assertEqual(
            sorted(path.name for path in (REPO_ROOT / "docker").glob("*deepseekv4*")),
            [],
        )
        workflow_text = (
            REPO_ROOT / ".github" / "workflows" / "pull-request.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("deepseekv4", workflow_text)

    def test_mindie_and_sglang_patches_live_in_dockerfiles(self):
        install_hook = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertNotIn('case "${UCM_ENGINE_TYPE:-}" in', install_hook)
        self.assertNotIn("boot_patch", install_hook)
        self.assertNotIn("sglang-adapt.patch", install_hook)

        mindie_dockerfile = (
            REPO_ROOT / "docker" / "Dockerfile.ucm-mindie-ascend.a2-v2"
        ).read_text(encoding="utf-8")
        self.assertIn("Apply patch for MindIE", mindie_dockerfile)
        self.assertIn("boot_patch", mindie_dockerfile)

        sglang_dockerfile = (
            REPO_ROOT / "docker" / "Dockerfile.ucm-sglang-cuda-v0.5.5"
        ).read_text(encoding="utf-8")
        self.assertIn("Apply patch for SGLang", sglang_dockerfile)
        self.assertIn("sglang-adapt.patch", sglang_dockerfile)

        sglang_a2_dockerfile = (
            REPO_ROOT / "docker" / "Dockerfile.ucm-sglang-ascend.a2-v0.5.18"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/build_ascend.sh", sglang_a2_dockerfile)
        self.assertNotIn("sglang-adapt.patch", sglang_a2_dockerfile)

    def test_build_scripts_package_install_hook(self):
        build_scripts = sorted((REPO_ROOT / "scripts").glob("build_*.sh"))

        self.assertEqual(len(build_scripts), 4)
        for path in build_scripts:
            text = path.read_text(encoding="utf-8")
            self.assertIn('cp "${KVCACHE_PROJECT_ROOT}/install.sh" .', text, path)
            self.assertIn("rm -f install.sh", text, path)

    def test_root_install_hook_documents_post_wheel_customization(self):
        installer = REPO_ROOT / "install.sh"
        text = installer.read_text(encoding="utf-8")

        self.assertIn("post-wheel", text)
        self.assertIn("custom installation steps", text)
        self.assertFalse((REPO_ROOT / "docker" / "install_ucm_wheel.sh").exists())


if __name__ == "__main__":
    unittest.main()
