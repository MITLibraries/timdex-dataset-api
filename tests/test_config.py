from timdex_dataset_api.config import configure_logger


def test_configure_logger_default_info_level(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    logger = configure_logger(__name__)

    info_msg = "hello INFO world"
    logger.info(info_msg)
    assert info_msg in caplog.text

    debug_msg = "hello DEBUG world"
    logger.debug(debug_msg)
    assert debug_msg not in caplog.text  # NOT captured


def test_configure_logger_env_var_sets_debug_level(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    monkeypatch.setenv("TDA_LOG_LEVEL", "DEBUG")
    logger = configure_logger(__name__)

    info_msg = "hello INFO world"
    logger.info(info_msg)
    assert info_msg in caplog.text

    debug_msg = "hello DEBUG world"
    logger.debug(debug_msg)
    assert debug_msg in caplog.text  # IS captured
