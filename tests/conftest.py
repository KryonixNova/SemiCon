import pytest


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A real (untrained, randomly-initialized) DriftSenseLocalizer
    checkpoint, session-scoped so building the encoder (a real ResNet-18)
    only happens once across the whole test run. Untrained weights are
    fine here -- these tests only check inference plumbing (shapes, CSV
    columns, error handling), never prediction accuracy."""
    import torch

    from src.localizer.config import LocalizerConfig
    from src.localizer.model import DriftSenseLocalizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DriftSenseLocalizer(LocalizerConfig()).to(device)
    align = model.calibrate()
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model": model.state_dict(), "align_offset": align,
               "step": 0, "metrics": {"acc@50px": 0.0}}, path)
    return str(path), device
