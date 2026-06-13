def test_fourth_down_package_imports():
    from model_training import fourth_down

    assert hasattr(fourth_down, "__version__")
