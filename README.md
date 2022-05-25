# NLP-project1 (fk58d7avc9)

## Evaluation

For access to the books please contact me at: [Erik Kastelec](mailto:erikkastelec@gmail.com)

* put txt files for corpus books into books directory
* install requirements

```
pip install -r ./ECKG/src/requirements.txt
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

#### Corpus

[Corpus.tsv](./books/corpus.tsv)
[Books](./books/) <br>

#### Entity co-occurrence graph (ECG)

[Helper functions (deduplication, graph building, entity importanc evaluation](./ECKG/src/helper_functions.py) <br>
[SVO triplet extraction](./ECKS/src/eventify.py)

