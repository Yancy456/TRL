from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "lvwerra/gpt2-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text with top_k sampling
output = model.generate(input_ids, max_length=50, top_k=50)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
