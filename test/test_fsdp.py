
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir='/scr-ssd/chenyuz/.cache')
peft_config = LoraConfig(
    inference_mode=False, r=8,
    lora_alpha=32,
    lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# this setup is necessary to use FSDP with PEFT (https://huggingface.co/docs/peft/en/accelerate/fsdp)
if getattr(accelerator.state, "fsdp_plugin", None) is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

model = accelerator.prepare(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer = accelerator.prepare(optimizer)

output_dir = "epoch_x"
accelerator.wait_for_everyone()
accelerator.save_state(output_dir)

