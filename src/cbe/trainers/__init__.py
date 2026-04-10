"""Trainer wrappers for different frameworks."""


def create_trainer(config, logger, artifact_store):
    """Create a framework-specific trainer from config."""
    if config.framework == "kauldron":
        from cbe.trainers.kd_trainer import KauldronTrainer
        return KauldronTrainer(config, logger=logger, artifact_store=artifact_store)
    elif config.framework == "huggingface":
        from cbe.trainers.hf_trainer import HFTrainer
        return HFTrainer(config, logger=logger, artifact_store=artifact_store)
    else:
        raise ValueError(f"Unknown framework: {config.framework}")
