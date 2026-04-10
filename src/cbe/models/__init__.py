"""Model factory utilities."""


def create_model(config):
    """Create a framework-specific model from config."""
    if config.framework == "kauldron":
        from cbe.models.kd_models import create_kd_model
        return create_kd_model(config.model)
    elif config.framework == "huggingface":
        from cbe.models.hf_models import create_hf_model
        return create_hf_model(config.model)
    else:
        raise ValueError(f"Unknown framework: {config.framework}")
