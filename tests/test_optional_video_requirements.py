# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> list[Requirement]:
    return [
        Requirement(line)
        for raw_line in path.read_text().splitlines()
        if (line := raw_line.split("#", 1)[0].strip()) and not line.startswith("-r ")
    ]


def _opencv_requirement(requirements: list[Requirement], machine: str) -> Requirement:
    environment = default_environment() | {"platform_machine": machine}
    matches = [
        requirement
        for requirement in requirements
        if requirement.name == "opencv-python-headless"
        and (requirement.marker is None or requirement.marker.evaluate(environment))
    ]
    assert len(matches) == 1
    return matches[0]


def test_common_requirements_do_not_install_opencv():
    requirements = _requirements(ROOT / "requirements/common.txt")
    names = {canonicalize_name(requirement.name) for requirement in requirements}
    assert "opencv-python-headless" not in names
    mistral_common = next(
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "mistral-common"
    )
    assert not mistral_common.extras


def test_video_extra_has_compatible_aarch64_opencv_pin():
    requirements = _requirements(ROOT / "requirements/video.txt")

    mistral_common = next(
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "mistral-common"
    )
    assert mistral_common.extras == {"image"}
    assert str(_opencv_requirement(requirements, "aarch64").specifier) == "==4.11.0.86"
    assert str(_opencv_requirement(requirements, "x86_64").specifier) == ">=4.13.0"


def test_setup_exposes_video_requirements_file():
    setup = (ROOT / "setup.py").read_text()
    assert '"video": _read_requirements("video.txt")' in setup
