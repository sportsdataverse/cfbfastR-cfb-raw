def test_package_imports():
    import model_training
    assert hasattr(model_training, "__version__")
