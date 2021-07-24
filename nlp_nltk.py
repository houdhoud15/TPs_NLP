# -*- coding: utf-8 -*-
import codecs
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import treebank
#mytext = """ربما كانت أحد أهم التطورات التي قامت بها الرياضيات العربية التي بدأت في هذا الوقت بعمل الخوارزمي و هي بدايات الجبر، و من المهم فهم كيف كانت هذه الفكرة الجديدة مهمة، فقد كانت خطوة ثورية بعيدا عن المفهوم اليوناني للرياضيات التي هي في جوهرها هندسة، الجبر كان نظرية موحدة تتيح الأعداد الكسرية و الأعداد اللا كسرية، و قدم وسيلة للتنمية في هذا الموضوع مستقبلا. و جانب آخر مهم لإدخال أفكار الجبر و هو أنه سمح بتطبيق الرياضيات على نفسها بطريقة لم تحدث من قبل"""
mytext = """Perhaps one of the most significant advances made byArabic mathematicsbegan at this time with the work of al-Khwarizmi, namely the beginnings of algebra. It is important to understand just how significant this new idea was. It was a revolutionary move away from the Greek concept of mathematics which was essentially geometry. Algebra was a unifying theory which allowedrational numbers,irrational numbers, geometrical magnitudes, etc., to all be treated as "algebraic objects". It gave mathematics a whole new development path so much broader in concept to that which had existed before, and provided a vehicle for future development of the subject. Another important aspect of the introduction of algebraic ideas was that it allowed mathematics to be applied to itselfin a way which had not happened before."""
words = word_tokenize(mytext)
phrase = sent_tokenize(mytext)
print("\nsent_tokenize : ", phrase)
print("#" * 50)
print("\nword_tokenize : ", words)
stopWords = set(stopwords.words('english'))
wordsFiltered = []
lemmatizing = []
stemming = []

for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
        lemmatizing.append(WordNetLemmatizer().lemmatize(w))
        stemming.append(PorterStemmer().stem(w))
tagg = pos_tag(wordsFiltered)
print("#" * 50)
print(tagg)
print("#" * 50)
print(nltk.chunk.ne_chunk(tagg))
print("#" * 50)
print(lemmatizing)
print("#" * 50)
print(stemming)
print("#" * 50)
#tree_bank = treebank.parsed_sents('wsj_0001.mrg')[0]
# tree_bank.draw()
