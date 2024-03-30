import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

cache_dir = '/scr-ssd/chenyuz/.cache/'
# model_id = "mistralai/Mistral-7B-v0.1"
# model_id = "mistralai/Mixtral-8x7B-v0.1"
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# config_ = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
# config_.load_in_4bit = True
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(model_id, config=config_, cache_dir=cache_dir) # load_in_4bit=True)
print(model)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def tokenize(text):
    return tokenizer.encode(text, add_special_tokens=False)

print(tokenizer.bos_token)
print(tokenizer.eos_token)
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id

PROMPT = 'Rate how humorous the given response is with a single number, on a scale of 1 to 5.'
USER_MESSAGE_1 = 'QUESTION: Tell me a joke. RESPONSE: Why did the scarecrow win an award? Because he was outstanding in his field. HUMOR SCORE:'
BOT_MESSAGE_1 = '4.'
USER_MESSAGE_2 = 'QUESTION: Say something. RESPONSE: I find the best way to start a conversation is with a joke. HUMOR SCORE:'
BOT_MESSAGE_2 = '2.'
USER_MESSAGE = 'QUESTION: How are you. RESPONSE: Hello, I am fine. HUMOR SCORE:'

USER_MESSAGE_x = PROMPT + f"""QUESTION: Does anybody feel like academic publication pressure is becoming unsustainable? I am becoming very frustrated with the publication culture in my field. Becoming an expert takes a long time and so is making a valuable contribution to the literature.  However, publication pressure is turning many contributions into spin-offs that are slightly different from the publication before, and they are often redundant. Further, a failed experiment would never get published but it would actually provide insight to peers as to what route not to explore. I think that publication pressure is overwhelming for academics and in detriment of scientific literature.  I feel like we seriously need to rethink the publication reward system.  Does anybody have thoughts on this? RESPONSE: Publishing a negative result requires some work, but it\'s definitely doable and I recommend it. Towards the beginning of my career I published two papers that are mostly "*if you think this is a good idea, don\'t; here\'s why*" and "*why XYZ looks like it should work, but doesn\'t*". Neither exceeded a dozen citations or so, but at least I contributed towards fighting the bias.  I deplore papers that are only slightly different spins on the same thing and I pity people who publish them (because otherwise they would perish). Like "*Novel method applied to material 1*". "*Novel method applied to material 2, which is very much like material 1*". Ugh. HUMOR SCORE:"""



instr = [BOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_1) + [EOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE_2) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_2) + [EOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE) + tokenize("[/INST]")

instr_x = [BOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_1) + [EOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE_2) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_2) + [EOS_ID] + \
    tokenize("[INST]") + tokenize(USER_MESSAGE_x) + tokenize("[/INST]")

# instruction = torch.tensor(instr).unsqueeze(0).to(0)
# outputs = model.generate(instruction, max_new_tokens=20)
# res = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(res)

# res = res.split("[/INST]")[-1]
# import re
# x = re.findall('[1-5]', res)[0]
# print(x)

# test batch inference
instructions = [torch.tensor(instr), torch.tensor(instr_x)]

# need attention mask
attention_mask = [torch.ones_like(ins) for ins in instructions]

instructions = torch.nn.utils.rnn.pad_sequence(
    instructions, batch_first=True, padding_value=EOS_ID).to(0)
print(instructions)
attention_mask = torch.nn.utils.rnn.pad_sequence(
    attention_mask, batch_first=True, padding_value=0).to(0)
print(attention_mask)

outputs = model.generate(instructions, attention_mask=attention_mask, max_new_tokens=20)
preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(preds)
