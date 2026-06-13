def test_fourth_down_package_imports():
    from model_training import fourth_down

    assert hasattr(fourth_down, "__version__")


def test_re_exports():
    from model_training import fourth_down as fd

    assert hasattr(fd, "FD_FEATURES")
    assert hasattr(fd, "FD_PARAMS")
    assert hasattr(fd, "fd_features")
    assert hasattr(fd, "train_fourth_down")
    assert hasattr(fd, "train_from_plays")
