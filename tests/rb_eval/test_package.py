def test_package_imports():
    import rb_eval

    assert hasattr(rb_eval, "__version__")
