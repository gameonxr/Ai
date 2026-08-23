from body import BodyLoader


def test_body_has_expected_dof():
    body = BodyLoader.load("config/body_humanoid.yaml")
    assert body.dof >= 12
    assert len(body.links) >= 10
    assert body.joints["neck"].range_radians[1] > 0
