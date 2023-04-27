from typing import Optional

from xturing.engines.new_engine import (
    Cerebras67bLoraInt8Engine,
    JapaneseGPT2LoraInt8Engine,
)
from xturing.models.causal import CausalLoraInt8Model


class Cerebras67bLoraInt8Model(CausalLoraInt8Model):
    config_name: str = "cerebras6.7b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Cerebras67bLoraInt8Engine.config_name, weights_path)


class JapaneseGPT2LoraInt8Model(CausalLoraInt8Model):
    config_name: str = "japanese_gpt2_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(JapaneseGPT2LoraInt8Engine.config_name, weights_path)
