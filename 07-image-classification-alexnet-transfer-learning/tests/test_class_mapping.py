import numpy as np
from src.class_mapping import map_original_labels

def test_mapping():
    assert map_original_labels(np.arange(10)).tolist()==[2,3,1,0,1,0,1,1,2,3]
