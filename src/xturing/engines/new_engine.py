from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalLoraEngine


class Cerebras67bLoraInt8Engine(CausalLoraEngine):
    config_name: str = "cerebras6.7b_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="cerebras/Cerebras-GPT-6.7B",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["c_attn"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class JapaneseGPT2LoraInt8Engine(CausalLoraEngine):
    config_name: str = "japanese_gpt2_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="rinna/japanese-gpt-1b",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["c_attn"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
