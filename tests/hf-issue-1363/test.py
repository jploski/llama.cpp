from transformers import GPT2TokenizerFast, GPT2Tokenizer

input_str = "2000"

tokenizer = GPT2Tokenizer(vocab_file='vocab.json', merges_file='merges.txt')
token_ids = tokenizer(input_str)["input_ids"]
print(token_ids, [ tokenizer.decode(x) for x in token_ids ])

tokenizer = GPT2TokenizerFast(vocab_file='vocab.json', merges_file='merges.txt', tokenizer_file='tokenizer.json')
token_ids = tokenizer(input_str)["input_ids"]
print(token_ids, [ tokenizer.decode(x) for x in token_ids ])
