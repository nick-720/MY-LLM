chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")


## unicode1

# A: What Unicode character does chr(0) return?
    a "null" character
# B: How does its repr differ from its printed form? You've seen this distinction before, on day one.
    the repr shows the null character. the printed form is an invisible character.
# C: What happens when this character occurs in text? Compare lines 3 and 4 closely. What's in the string, versus what you see?
    if I don't use print it displays the null character. If it's contained in a print statement it becomes invisible. the null character is always there, it's just hidden with a print statement. 

## unicode2

# A: Why train on UTF-8 rather than UTF-16 or UTF-32?
    UTF-8 uses fewer bytes for english text. a byte level tokenizer would use far too many tokens for UTF-16 and UTF-32 because english encodings use padding bytes between characters. However, UTF-16  can be more token efficient for some foreign languages, while UTF-32 can never since it spends four bytes per character.
# B: The broken decoder. Find the input that breaks this function: 
#    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    decode_utf8_bytes_to_str_wrong("héllo".encode("utf-8")) breaks the function because the function treats every character as one byte. and in this case, é is more than one byte.

# C: try b'\xa9\xc3'.decode("utf-8") and work out why the same two bytes are valid in one order and garbage in the other.
    b'\xa9\xc3' doesn't work because certain bytes have certain roles in sequences. 0xa9 can't start a sequence in UTF-8, 0xc3 can. 0xc3 is a start byte so it will be invalid if there's no subsequent bytes. 


## pre-tokenization
# A: What is the longest token the tokenizer learns on the training text, and why?
    b' accomplishment' because the longest token is the longest string that appears often enough to win all of its merge. in TinyStories, 'accomplishment' is unusually common for a long word because of it's recurring use in phrases at the end of a story.