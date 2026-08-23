from simulator import ConfigLoader


def test_config_loader_resolves_references():
    config = ConfigLoader("config/simulator_config.yaml").load()
    assert config["body"]["loaded"]["body"]["type"] == "humanoid"
    assert config["physics"]["loaded"]["physics"]["gravity"][2] == -9.81
