# fasttext-demo

- Install fasttext

`git clone https://github.com/facebookresearch/fastText.git`

`cd fastText`

`sudo pip install .`

- Download 1Gb Wikipedia corpus

`curl http://mattmahoney.net/dc/enwik9.zip --output enwik9.zip`

`unzip data/enwik9.zip -d data`

- Download Cooking stack exchange corpus

`curl https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz --output cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz`

`cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt`

`head -n 12404 cooking.preprocessed.txt > cooking.train`

`tail -n 3000 cooking.preprocessed.txt > cooking.valid`
