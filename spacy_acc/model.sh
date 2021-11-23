#!usr/bin/bash
cd $(dirname $0)
python3 -m spacy train config.cfg --output ./output --paths.train ./trainhier.spacy --paths.dev ./devhier.spacy && spacy evaluate ./output/model-best ./testhier.spacy > scores.txt
