import re

import rwls


def test_version_is_a_release_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+", rwls.__version__)


def test_installed_version_matches_the_package():
    from importlib.metadata import version

    assert version("rwls") == rwls.__version__
