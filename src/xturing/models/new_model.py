from typing import Optional

from xturing.engines.new_engine import Calm7bLoraInt8Engine, Cerebras67bLoraInt8Engine
from xturing.models.causal import CausalLoraInt8Model


class Cerebras67bLoraInt8Model(CausalLoraInt8Model):
    config_name: str = "cerebras6_7b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Cerebras67bLoraInt8Engine.config_name, weights_path)


class Calm7bLoraInt8Model(CausalLoraInt8Model):
    config_name: str = "calm7b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Calm7bLoraInt8Engine.config_name, weights_path)
