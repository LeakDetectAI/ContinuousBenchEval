"""Data loading and formatting utilities."""

from cbe.data.formatters import load_jsonl, clean_text


def create_data_pipeline(config):
    """Create a framework-specific data pipeline from config."""
    if config.framework == "kauldron":
        from cbe.data.kd_data import KauldronDataPipeline
        return KauldronDataPipeline(config)
    elif config.framework == "huggingface":
        from cbe.data.hf_data import HFDataPipeline
        return HFDataPipeline(config)
    else:
        raise ValueError(f"Unknown framework: {config.framework}")
