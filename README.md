# NLP-project1 (fk58d7avc9)

## Evaluation

For access to the books please contact me at: [Erik Kastelec](mailto:erikkastelec@gmail.com)

* put txt files for corpus books into books directory
* install requirements

```
pip install -r ./ECKG/requirements.txt
pip install git+https://github.com/hltcoe/PredPatt.git
python -m spacy download en_core_web_trf
```

To run literary character recognition and ranking run:

```
python ./ECKG/src/evaluation.py
```

To run relation detection and protagonist/antagonist prediction run:

```
python ./ECKG/src/character_eval.py
```

For comparison of different relation detection and protagonist/antagonist prediction run:
```
python ./ECKG/src/eval_corpus.py
```


#### Corpus

[Corpus.tsv](./books/corpus.tsv)


