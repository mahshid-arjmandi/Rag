import numpy as np
import nltk
import re
import string
import warnings
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from configparser import ConfigParser
from functools import reduce
from gensim.models import Doc2Vec
#from hazm.Embedding import SentEmbedding
from hazm import *
from transformers import AutoTokenizer




keyword_count = 10


# مسیر محلی مدل
local_model_path = r"E:\University\master\mbaheseVijeh\project_AI\Rag\Models"

# بارگذاری توکنایزر از مسیر محلی
try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
except Exception as e:
    print("خطا در بارگذاری توکنایزر از مسیر محلی:", e)
    exit()


import os

# مرحله 0: دریافت مسیر فایل متنی
text_file_path = r"E:\University\master\mbaheseVijeh\Text.txt"

# بررسی وجود فایل
if not os.path.exists(text_file_path):
    print("فایل وجود ندارد. لطفاً مسیر درست وارد کنید.")
    exit()

# خواندن محتویات فایل متنی
try:
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except Exception as e:
    print("خطا در خواندن فایل:", e)
    exit()

# ادامه اجرای کد با متن دریافتی
print("متن دریافت شده از فایل:")
print(text)

# متن فارسی پیچیده
#text = 'سفارت ایران در مادرید درباره فیلم منتشرشده از «حسن قشقاوی» در مراسم سال نو در کاخ سلطنتی اسپانیا و حاشیه‌سازی‌ها در فضای مجازی اعلام کرد: به تشریفات دربار کتباً اعلام شد سفیر بدون همراه در مراسم حضور خواهد داشت و همچون قبل به دلایل تشریفاتی نمی‌تواند با ملکه دست بدهد. همان‌گونه که کارشناس رسمی تشریفات در توضیحات خود به یک نشریه اسپانیایی گفت این موضوع توضیح مذهبی داشته و هرگز به معنی بی‌احترامی به مقام و شخصیت زن آن هم در سطح ملکه محترمه یک کشور نیست.'

# مرحله 1: نرمال‌سازی متن
normalizer = Normalizer()
normalized_text = normalizer.normalize(text)

# حذف نیم‌فاصله‌ها
normalized_text = normalized_text.replace('\u200c', '')

# مرحله 2: تقسیم متن به جملات
sentences = sent_tokenize(normalized_text)

# مرحله 3: توکنایز کردن و بازسازی توکن‌ها
updated_tokenized_text = [] # لیست توکن‌های بازسازی‌شده
reconstructed_sentences = [] # لیست جملات بازسازی‌شده

for sentence in sentences:
    tokens = tokenizer.tokenize(sentence)
    updated_tokens = []
    reconstructed_sentence = ""

    for token in tokens:
        if token.startswith("##"):
            # اتصال توکن با ## به قبلی (هم در لیست توکن‌ها و هم در جمله)
            updated_tokens[-1] += token[2:]
            reconstructed_sentence += token[2:]
        else:
            # اضافه کردن توکن جدید
            updated_tokens.append(token)
            if reconstructed_sentence: # اضافه کردن فاصله قبل از توکن جدید
                reconstructed_sentence += " "
            reconstructed_sentence += token

    updated_tokenized_text.append(updated_tokens)
    reconstructed_sentences.append(reconstructed_sentence.strip())

# نمایش نتایج
print("Normalized Text:")
print(normalized_text)
print("\nUpdated Tokenized Text:")
#for i, tokens in enumerate(updated_tokenized_text):
#    print(f"Sentence {i + 1} Tokens: {' '.join(tokens)}")
print(updated_tokenized_text)
print("\nReconstructed Sentences:")
#for i, sentence in enumerate(reconstructed_sentences):
#    print(f"Sentence {i + 1}: {sentence}")
print(reconstructed_sentences)



model_path = r'E:\University\master\mbaheseVijeh\project_AI\Rag\POST_Tagger\pos_tagger.model'
tagger = POSTagger(model = model_path)
#token_tag_list = tagger.tag_sents(tokenize_text)
token_tag_list = tagger.tag_sents(updated_tokenized_text)
print('===================================','\n',token_tag_list)

grammers = [
"""
NP:
        {<NOUN,EZ>?<NOUN.*>}    # Noun(s) + Noun(optional)

""",

"""
NP:
        {<NOUN.*><ADJ.*>?}    # Noun(s) + Adjective(optional)

"""
]
## you can also add your own grammer to be extracted from the text...

def extract_candidates(tagged, grammer):
    keyphrase_candidate = set()
    np_parser = nltk.RegexpParser(grammer)
    trees = np_parser.parse_sents(tagged)
    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))
    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}
    keyphrase_candidate = list(keyphrase_candidate)
    return keyphrase_candidate

all_candidates = set()
for grammer in grammers:
    all_candidates.update(extract_candidates(token_tag_list, grammer))


all_candidates = np.array(list(all_candidates))


print('==========================','\n',np.array(list(all_candidates)))

sent2vec_model_path = r'E:\University\master\mbaheseVijeh\project_AI\Rag\Sen2Vec\sent2vec-naab.model'
sent2vec_model = SentEmbedding(sent2vec_model_path)
#print(sent2vec_model)

all_candidates_vectors = [sent2vec_model[candidate] for candidate in all_candidates]
all_candidates_vectors[0:2]

#print("\n",'=========',all_candidates_vectors[0:2])


candidates_concatinate = ' '.join(all_candidates)
whole_text_vector = sent2vec_model[candidates_concatinate]
whole_text_vector


candidates_sim_whole = cosine_similarity(all_candidates_vectors, whole_text_vector.reshape(1,-1))
candidates_sim_whole.reshape(1,-1)






candidate_sim_candidate = cosine_similarity(all_candidates_vectors)
candidate_sim_candidate


candidates_sim_whole_norm = candidates_sim_whole / np.max(candidates_sim_whole)
candidates_sim_whole_norm = 0.5 + (candidates_sim_whole_norm - np.average(candidates_sim_whole_norm)) / np.std(candidates_sim_whole_norm)
candidates_sim_whole_norm

#print('\n','======',candidates_sim_whole_norm)


np.fill_diagonal(candidate_sim_candidate, np.NaN)
candidate_sim_candidate_norm = candidate_sim_candidate / np.nanmax(candidate_sim_candidate, axis=0)
candidate_sim_candidate_norm = 0.5 + (candidate_sim_candidate_norm - np.nanmean(candidate_sim_candidate_norm, axis=0)) / np.nanstd(candidate_sim_candidate_norm, axis=0)
candidate_sim_candidate_norm
#print('\n','========',candidate_sim_candidate_norm)

beta = 0.82
N = min(len(all_candidates), keyword_count)

selected_candidates = []
unselected_candidates = [i for i in range(len(all_candidates))]
best_candidate = np.argmax(candidates_sim_whole_norm)
selected_candidates.append(best_candidate)
unselected_candidates.remove(best_candidate)


for i in range(N-1):
    selected_vec = np.array(selected_candidates)
    unselected_vec = np.array(unselected_candidates)

    unselected_candidate_sim_whole_norm = candidates_sim_whole_norm[unselected_vec, :]

    dist_between = candidate_sim_candidate_norm[unselected_vec][:, selected_vec]

    if dist_between.ndim == 1:
        dist_between = dist_between[:, np.newaxis]

    best_candidate = np.argmax(beta * unselected_candidate_sim_whole_norm - (1 - beta) * np.max(dist_between, axis = 1).reshape(-1,1))
    best_index = unselected_candidates[best_candidate]
    selected_candidates.append(best_index)
    unselected_candidates.remove(best_index)
all_candidates[selected_candidates].tolist()
print(all_candidates[selected_candidates].tolist())

# مسیر فایل خروجی برای ذخیره کلمات کلیدی
keywords_file_path =r"E:\University\master\mbaheseVijeh\keywords.txt"

# استخراج کلمات کلیدی
keywords = all_candidates[selected_candidates].tolist()

# ذخیره کلمات کلیدی در فایل متنی
try:
    with open(keywords_file_path , 'w', encoding='utf-8') as file:
        for keyword in keywords:
            file.write(keyword + '\n')
    print(f"کلمات کلیدی با موفقیت در فایل {keywords_file_path } ذخیره شدند.")
except Exception as e:
    print("خطا در ذخیره فایل:", e)
