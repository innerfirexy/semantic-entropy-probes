import torch
from uncertainty.utils import utils
from transformers import StoppingCriteriaList

parser = utils.get_parser()
args, unknown = parser.parse_known_args()

print(args.model_name)

model = utils.init_model(args)
model.model.to("cpu")

input_data = 'Hello, how are you?'
inputs = model.tokenizer(input_data, return_tensors="pt").to("cpu")
pad_token_id = model.tokenizer.eos_token_id

temperature = 0.1 


with torch.no_grad():
    outputs = model.model.generate(
        **inputs,
        max_new_tokens=model.max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
        temperature=temperature,
        do_sample=False,
        stopping_criteria=None,
        pad_token_id=pad_token_id,
    )

print()
full_answer = model.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)
print(full_answer)

print('***************')

# transition scores
transition_scores = model.model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True)
print(transition_scores) # The transition scores are only non-zero when do_sample is False, otherwise they are (almost) all zeros.