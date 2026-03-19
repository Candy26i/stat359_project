from arithmetic_tokenizer import ArithmeticBPETokenizer

s1 = "((((14 - 13) + (15 - 5)) + 19) + (2 - ((10 + 17) - (18 + 11)))) + (9 - (((19 - 7) + (8 + 6)) + 7))"
s2 = "14 + 3"
s3 = "18 - (20 + (14 + ((15 + 16) - 2 + 1)))"
s4 = "14 + (3 + 1) - (10 + 10)"
s5 = """
Evaluate : ( ( 19 - ( ( 16 + 14 ) + 19 ) ) + ( ( ( 7 - 1 ) + ( 10 - 9 ) ) - ( ( 11 - 14 ) - ( 11 - 4 ) ) ) ) - ( 18 - ( ( ( 7 - 1 ) + ( 13 + 3 ) ) + ( ( 4 - 15 ) + 19 ) ) ) <think> <think> Step 1 : 16 + 14 = 30 Expression now : ( ( 19 - ( 16 + 19 ) ) + ( ( 7 - 1 ) + ( 10 - 9 ) ) ) - ( ( 11 - 14 ) - ( 11 - 4 ) ) Step 2 : 16 + 19 = 34 Expression now : ( ( 19 - 34 ) + ( ( 7 - 1 ) + ( 10 - 9 ) ) ) - ( ( 11 - 14 ) - ( 11 - 4 ) ) Step 3 : 19 - 34 = - 16 Expression now : ( - 16 + ( ( 7 - 1 ) + ( 10 - 9 ) ) ) - ( ( 11 - 14 ) - ( 11 - 4 ) ) Step 4 : 7 - 1 = 6 Expression now : ( - 16 + ( 6 + ( 10 - 9 ) ) ) - ( ( 11 - 14 ) - ( 11 -
"""
s5_query = "Evaluate : ( ( 19 - ( ( 16 + 14 ) + 19 ) ) + ( ( ( 7 - 1 ) + ( 10 - 9 ) ) - ( ( 11 - 14 ) - ( 11 - 4 ) ) ) ) - ( 18 - ( ( ( 7 - 1 ) + ( 13 + 3 ) ) + ( ( 4 - 15 ) + 19 ) ) )"

s6 = "1986"
s7 = "2222"
tok = ArithmeticBPETokenizer()
tok.load("C:\\Users\\madis\\Desktop\\stat_359\\stat359\\data\\tokenizer")

t1 = tok.encode(s1, add_special_tokens=True)
t2 = tok.encode(s2, add_special_tokens=True)
t3 = tok.encode(s3, add_special_tokens=True)
t4 = tok.encode(s4, add_special_tokens=True)
t5 = tok.encode(s5, add_special_tokens=True)
t5_query = tok.encode(s5_query, add_special_tokens=True)
t6 = tok.encode(s6, add_special_tokens=True)
t7 = tok.encode(s7, add_special_tokens=True)
print("len1 =", len(t1))
print("len2 =", len(t2))
print("len3 =", len(t3))
print("len4 =", len(t4))
print("len5 =", len(t5))
print("len5_query =", len(t5_query))
print("len6 =", len(t6))
print("len7 =", len(t7))
# 打印前后若干 token id
print("t1 head:", t1[:60])
print("t1 tail:", t1[-60:])
print("t2 head:", t2[:60])
print("t2 tail:", t2[-60:])
print("t3 head:", t3[:60])
print("t3 tail:", t3[-60:])
print("t4 head:", t4[:60])
print("t4 tail:", t4[-60:])
print("t5 head:", t5[:60])
print("t5 tail:", t5[-60:])
print("t5_query head:", t5_query[:60])
print("t5_query tail:", t5_query[-60:])
print("t6 head:", t6[:60])
print("t6 tail:", t6[-60:])
print("t7 head:", t7[:60])
print("t7 tail:", t7[-60:])
# 如果支持 decode，看看 tokenization 还原是否一致
if hasattr(tok, "decode"):
    print("decode1 preview:", tok.decode(t1)[:200])
    print("decode2 preview:", tok.decode(t2)[:200])
    print("decode3 preview:", tok.decode(t3)[:200])
    print("decode4 preview:", tok.decode(t4)[:200])
    print("decode5 preview:", tok.decode(t5)[:200])
    print("decode5_query preview:", tok.decode(t5_query)[:200])
    print("decode6 preview:", tok.decode(t6)[:200])
    print("decode7 preview:", tok.decode(t7)[:200])