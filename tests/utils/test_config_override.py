import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src import utils


def test_use_config_override_returns_deepcopy_and_resets():
    baseline = utils.load_config()
    override = {
        "override_test": {
            "enabled": True,
            "values": ["a", "b"],
        }
    }

    with utils.use_config_override(override):
        cfg1 = utils.load_config()
        cfg2 = utils.load_config()

        assert cfg1 == override
        assert cfg2 == override

        cfg1["override_test"]["values"].append("c")
        assert utils.load_config()["override_test"]["values"] == ["a", "b"]

    assert utils.load_config() == baseline


def test_use_config_override_nested_scope_restores_outer_override():
    marker = "__config_override_nested_test_key__"

    with utils.use_config_override({marker: "outer"}):
        assert utils.load_config()[marker] == "outer"

        with utils.use_config_override({marker: "inner"}):
            assert utils.load_config()[marker] == "inner"

        assert utils.load_config()[marker] == "outer"

    assert marker not in utils.load_config()
