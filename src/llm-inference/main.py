from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)


print("Compile model for the NPU")
config = CompilerConfig(dtype=torch.float64)
model = intel_npu_acceleration_library.compile(model, config)

query = "What is the meaning of life?"
prefix = tokenizer(query, return_tensors="pt")["input_ids"]


generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
