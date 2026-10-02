import horizonchart


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("STARPLOT_DATA_PATH", str(tmp_path))
    assert horizonchart._data_path() == tmp_path


def test_shared_folder_used_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("STARPLOT_DATA_PATH", raising=False)
    monkeypatch.setattr(horizonchart, "SHARED_DATA_PATH", tmp_path)
    assert horizonchart._data_path() == tmp_path


def test_platform_cache_otherwise(monkeypatch, tmp_path):
    monkeypatch.delenv("STARPLOT_DATA_PATH", raising=False)
    monkeypatch.setattr(horizonchart, "SHARED_DATA_PATH", tmp_path / "missing")
    path = horizonchart._data_path()
    assert "horizonchart" in path.parts  # ...\horizonchart\Cache on Windows
    assert "missing" not in str(path)
