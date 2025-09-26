from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type

# refer to https://github.com/Yuanshi9815/OminiControl
class select_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], adapter_name) -> None:
        self.adapter_name =  adapter_name

        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            for active_adapter in lora_module.active_adapters:
                if active_adapter != self.adapter_name:
                    lora_module.scaling[active_adapter] = 0

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for lora_module in self.lora_modules:
            for active_adapter in lora_module.active_adapters:
                if active_adapter != self.adapter_name:
                    lora_module.scaling[active_adapter] = 1