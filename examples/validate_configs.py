import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_validation import ConfigurationValidator


if __name__ == "__main__":
    validator = ConfigurationValidator()
    paths = [Path("config/simulator_config.yaml")]
    reports = [validator.validate(path) for path in paths]
    for report in reports:
        status = "OK" if report.valid else "INVALID"
        print(f"{status}: {report.path}")
        for message in report.errors + report.warnings:
            print(f"  - {message}")
    if any(not report.valid for report in reports):
        raise SystemExit(1)
