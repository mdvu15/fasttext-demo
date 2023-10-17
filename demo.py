'''
- Install fasttext

git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .

- Download 1Gb Wikipedia corpus

curl http://mattmahoney.net/dc/enwik9.zip --output enwik9.zip
unzip data/enwik9.zip -d data

- Download Cooking stack exchange corpus

curl https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz --output cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
head -n 12404 cooking.preprocessed.txt > cooking.train
tail -n 3000 cooking.preprocessed.txt > cooking.valid

'''

# %%
import fasttext
import pprint
pp = pprint.PrettyPrinter(indent=4)

# %%
# model = fasttext.train_unsupervised('data/fil9')
# model.save_model("result/fil9.bin")
# %%
# model_amz = fasttext.train_unsupervised('data/archive/test.ft.txt')
# model_amz.save_model("result/fil_amz.bin")
# %%
# model_cooking = fasttext.train_supervised(input="data/cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
# model_cooking.save_model("result/cooking_preprocessed_model.bin")
# %%
# Load pre-trained models
model = fasttext.load_model("/Users/minhdvu/Projects/fastText/result/fil9.bin")
model_amz = fasttext.load_model("/Users/minhdvu/Projects/fastText/result/fil_amz.bin")
model_cooking = fasttext.load_model("/Users/minhdvu/Projects/fastText/result/cooking_preprocessed_model.bin")

# %%
# Get vector representation
model.get_word_vector('asparagus')

# %%
# Get nearest neighbors using different models
model.get_nearest_neighbors('texas')
# %%
model_amz.get_nearest_neighbors('texas')

# %%
# Get inferences
model.get_nearest_neighbors('pytho')
# %%
model.get_nearest_neighbors('javasscript')

# %%
# Get similar meaning/associated words
print('---Nearest Neighbors:---')
print("taste: ", end = "")
pp.pprint(model_cooking.get_nearest_neighbors("taste")[:7])
print("mushroom: ", end = "")
pp.pprint(model_cooking.get_nearest_neighbors("mushroom")[:7])
print("crispy: ", end = "")
pp.pprint(model_cooking.get_nearest_neighbors("crispy")[:7])
print()

# Get equivalent of a presented analogy
print('---Get Analogies:---')
print("chili of sweetness:", end = " ")
pp.pprint(model_cooking.get_analogies("chili", "spicy","sweet"))
print("brocolli of fruits:", end = " ")
pp.pprint(model_cooking.get_analogies("brocolli", "vegetable","fruit"))
print()

# Give label from query
print('---Label Prediction:---')
print(model_cooking.predict("Which dish is best to bake a banana bread ?"))
print(model_cooking.predict("How much spiciness is too much ?"))


# %%
# Custom function to label words based on word vectors
from numpy import dot
from numpy.linalg import norm

def label(labels, word):
    l1, l2 = labels
    l1_v, l2_v = model.get_word_vector(l1), model.get_word_vector(l2)
    word_v = model.get_word_vector(word)

    # Cosine similarity
    l1_sim = dot(l1_v, word_v)/(norm(l1_v)*norm(word_v))
    l2_sim = dot(l2_v, word_v)/(norm(l2_v)*norm(word_v))
    return l1 if l1_sim > l2_sim else l2

# %%
label(['messi', 'ronaldo'], 'goat')
# %%
label(['good', 'bad'], 'pineapple on pizza')
# %%
label(['positive', 'negative'], 'horrible')
# %%
label(['positive', 'negative'], 'wonderful')
# %%
languages = ['python', 'java', 'javascript', 'c++']
[label(['positive', 'negative'], lang) for lang in languages]
