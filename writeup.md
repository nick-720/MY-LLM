chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")


## unicode1

### A: What Unicode character does chr(0) return?
    a "null" character
### B: How does its repr differ from its printed form? You've seen this distinction before, on day one.
    the repr shows the null character. the printed form is an invisible character.
## C: What happens when this character occurs in text? Compare lines 3 and 4 closely. What's in the string, versus what you see?
    if I don't use print it displays the null character. If it's contained in a print statement it becomes invisible. the null character is always there, it's just hidden with a print statement. 

## unicode2

### A: Why train on UTF-8 rather than UTF-16 or UTF-32?
    UTF-8 uses fewer bytes for english text. a byte level tokenizer would use far too many tokens for UTF-16 and UTF-32 because english encodings use padding bytes between characters. However, UTF-16  can be more token efficient for some foreign languages, while UTF-32 can never since it spends four bytes per character.
### B: The broken decoder. Find the input that breaks this function: 
###    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
###        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    decode_utf8_bytes_to_str_wrong("héllo".encode("utf-8")) breaks the function because the function treats every character as one byte. and in this case, é is more than one byte.

# C: try b'\xa9\xc3'.decode("utf-8") and work out why the same two bytes are valid in one order and garbage in the other.
    b'\xa9\xc3' doesn't work because certain bytes have certain roles in sequences. 0xa9 can't start a sequence in UTF-8, 0xc3 can. 0xc3 is a start byte so it will be invalid if there's no subsequent bytes. 


## BPE Training on TinyStories (train_bpe_tinystories)
### A: How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Training took 61 seconds and peaked at about 7.8 GB of memory. Longest vocab token is b' accomplishment', which makes sense because BPE merges the most frequent pairs, and stories often end with phrases like 'a sense of accomplishment.'
### B: Profile your code. What part of the tokenizer training process takes the most time?
Merging takes the most time, about 35 seconds, roughly 35% more than parallelized pre-tokenization at 26 seconds. 

## BPE Training on OpenWebText (train_bpe_expts_owt)
### A: What is the longest token in the vocabulary? Does it make sense?
The longest token is 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ', which makes sense because of mojibake, a common phenomenon in scraped web text: characters like é become two characters if encoded with UTF-8 and decoded with Latin-1. When this happens repeatedly, you can get long sequences of those new characters, and the BPE tokenizer packs those sequences into a token because of their frequency.
### B: Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
TinyStories shares 72% of its learned tokens with OWT and TinyStories-only tokens are mostly friendly/casual since they come from children's stories, while OWT-only tokens are highly technical because they are scraped from general web text (tech, news, science, gaming, etc.). TinyStories has approx 60,000 distinct pre-tokens, and OWT has 6.6 million which explains why it has a broader vocabulary (junk included).

## Tokenizer Experiments
### (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained
### TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively),
### encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio
### (bytes/token)?
From each validation file, I split on `<|endoftext|>` and kept `docs[1:11]` (skipping the first piece in case it was cut off), and then dividing once the total encoded bytes across all documents by its total tokens generated we get the following results.
TinyStories compression ratio: 4.03
OpenWebText compression ratio: 4.51

### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?
### Compare the compression ratio and/or qualitatively describe what happens.
Tokenizing OWT through TinyStories tokenizer results in a lower compression ratio (4.51 vs. 3.44 = ~30% more tokens). This is because TinyStories never learned the words from OWT so the TinyStories tokenizer has to build more tokens from the OWT words, and because there are fewer merges to use.
### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
### tokenize the Pile dataset (825GB of text)?
Approximately 1.76 days: tested throughput on 10MB of OWT_valid on owt_vocab and owt_merges and got 5,411,000 bytes/second, scaling up to 825GB @ the same throughput gets us to 1.76 days. This assumes throughput stays constant. Heaps' law suggests are larger run would be somewhat faster, while the Pile's more varied sources could slow it down.

### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and
### development datasets into a sequence of integer token IDs. We’ll use this later to train our
### language model. We recommend serializing the token IDs as a NumPy array of datatype
### uint16. Why is uint16 an appropriate choice?
uint16 is appropriate because our largest ID is 31,999 (32k tokens in OWT vocab, 10k in TS vocab), and uint16 holds ~65.5k values. uint8 would be far too small (256 values) and the other alternative int64 holds more than enough, but each id would be 8 bytes so the file size would be unnecessarily large (21.6GB with int64 vs. 5.4GB with uint16 on OWT training set)