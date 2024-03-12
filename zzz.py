import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# plt.figure()
# x = torch.arange(1, 10000)
# xiao = ((x ** -.5) - (x * 1500 ** -1.5) <= 0).float()
# y = (768 ** -.5) * ((x ** -.5) * xiao + (x * 1500 ** -1.5) * (xiao == 0).float())
# plt.plot(x, y)
# plt.savefig('img.png')
from torchinfo import summary
from transformers import GPT2Tokenizer, GPT2Model
# model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
txt = 'i am a good boy.'
b = tokenizer(txt, return_tensors='pt')['input_ids']
a = tokenizer._tokenize(txt)
encoded_input = [tokenizer._convert_token_to_id(t) for t in tokenizer._tokenize(txt)]
# encoded_input = [tokenizer._convert_id_to_token(t) for t in encoded_input]
# output = model(**encoded_input)
print(1)


