from src.config import ExperimentConfig

def test_config():
    c=ExperimentConfig.load(); assert c.batch_size>0 and c.image_size==96 and "alexnet_style" in c.models
