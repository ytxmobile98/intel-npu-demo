# Source: https://intel.github.io/intel-npu-acceleration-library/index.html#run-a-llama-model-on-the-npu

from transformers import AutoTokenizer, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM
from intel_npu_acceleration_library.compiler import CompilerConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

config = CompilerConfig()
model = NPUModelForCausalLM.from_pretrained(
    model_id, config, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

query = input("Ask something: ")
prefix = tokenizer(query, return_tensors="pt")["input_ids"]

generation_kwargs = dict(
   input_ids=prefix,
   streamer=streamer,
   do_sample=True,
   top_k=50,
   top_p=0.9,
   max_new_tokens=512,
)

print("Run inference")
_ = model.generate(**generation_kwargs)
