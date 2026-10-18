import re

import rwls


def test_version_is_a_release_string():
    assert re.fullmatch(r"\d+\.\d+\.\d+", rwls.__version__)
