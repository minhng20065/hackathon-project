# split string at given character, return both pieces of string
ls = []
def split2(string, char):
    a = string.split(char, 1)[0]
    b = string.split(char, 1)[1]
    return a, b

# detect number of capital letters
def num_cap(string):
    return sum(1 for i in string if i.isupper())

spam_kword = ["free", "join now", "urgent", 
