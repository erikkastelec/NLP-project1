# SloCOREF - Coreference Resolution for Slovene language

V tem repozitoriju se nahaja rezultat aktivnosti A3.2 - R3.2.3 Orodje za odkrivanje koreferenčnosti, ki je nastalo v okviru projekta [Razvoj slovenščine v digitalnem okolju](https://slovenscina.eu).

---

Slovene coreference resolution project introduces four models for coreference resolution on coref149 and senticoref datasets:

- baseline model (linear regression with hand-crafted features),
- non-contextual neural model using word2vec embeddings,
- contextual neural model using ELMo embeddings,
- contextual neural model using BERT embeddings.

For more details, [see the journal paper](https://doi.org/10.2298/CSIS201120060K), published in Computer Science and Information Systems (2021).

BERT-based pre-trained model on [SentiCoref 1.0 dataset](https://www.clarin.si/repository/xmlui/handle/11356/1285) is available in [the project data repository](https://nas.cjvt.si/index.php/f/21605774).

## Project structure

- `report/` contains the pdf of our work.
- `src/` contains the source code of our work.
- `data/` is a placeholder for datasets (see _Getting datasets_ section below).

## Setup

Before doing anything, the dependencies need to be installed.  
```bash
$ pip install -r requirements.txt
```

**Note**: if you have problems with `torch` library, make sure you have python x64 installed. Also make use of 
[this](https://pytorch.org/get-started/locally/#start-locally) official command builder.

**Notes**: you might have problems running contextual models on Windows since [allennlp](https://github.com/allenai/allennlp#installation) is not officialy supported on Windows, as noted in their README.

### Getting datasets

The project operates with the following datasets: 
- [SSJ500k](https://www.clarin.si/repository/xmlui/handle/11356/1210) (`-sl.TEI` version), 
- [coref149](https://www.clarin.si/repository/xmlui/handle/11356/1182)
- [sentiCoref 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1285) ([WebAnno TSV 3.2 File format docs](https://zoidberg.ukp.informatik.tu-darmstadt.de/jenkins/job/WebAnno%20(GitHub)%20(master)/de.tudarmstadt.ukp.clarin.webanno$webanno-webapp/doclinks/1/#sect_webannotsv)).

Download and extract them into `data/` folder. After that, your data folder should look like this:
```
data/
+-- ssj500k-sl.TEI
    +-- ssj500k.back.xml
    +-- ssj500k-sl.body.xml
    +-- ssj500k-sl.xml
    +-- ...
+-- coref149
    +-- ssj4.15.tcf
    +-- ssj5.30.tcf
    +-- ... (list of .tcf files)
+-- senticoref1_0
    +-- 1.tsv
    +-- 2.tsv
    +-- ... (list of .tsv files)
```

Coref149 and SentiCoref are the main datasets we use. 

SSJ500k is used for additional metadata such as dependencies and POS tags, which are not provided by coref149 itself.

Since only a subset of SSJ500k is used, it can be trimmed to decrease its size and improve loading time. 
To do that, run `trim_ssj.py`:
```bash
$ python src/trim_ssj.py --coref149_dir=data/coref149 --ssj500k_path=data/ssj500k-sl.TEI/ssj500k-sl.body.xml --target_path=data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml
```

If `target_path` parameter is not provided, the above command would produce 
`data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml`.

If you want to use pretrained embeddings in non-contextual coreference model, make sure to download the Slovene
- word2vec vectors (`Word2Vec Continuous Skipgram`) from http://vectors.nlpl.eu/repository/ and/or
- fastText vectors (`bin`) from https://fasttext.cc/docs/en/crawl-vectors.html (not used in the paper but supported)

Put them into `data/` (either the `cc.sl.300.bin` file for fastText or the `model.txt` file for word2vec).

For the contextual coreference model, make sure to download [the pretrained Slovene ELMo embeddings](https://www.clarin.si/repository/xmlui/handle/11356/1277). 
Extract the options file and the weight file into `data/slovenian-elmo`.


## Running the project

Before running anything, make sure to set `DATA_DIR` and `SSJ_PATH` parameters in `src/data.py` file (if the paths to 
where your datasets are stored differ in your setup).

Below are examples how to run each model.

Parameters and it's default values can be previewed at the top of each model's file.

`--dataset` parameter can be either `coref149` or `senticoref`.

### Baseline model

Baseline model (linear regression with hand-crafted features)

```bash
$ python baseline.py \
  --model_name="my_baseline_model" \
  --learning_rate="0.05" \
  --dataset="coref149" \
  --num_epochs="20" \
  --fixed_split
```

### Non-contextual model

Non-contextual_model with word2vec embeddings

```bash
$ python noncontextual_model.py \
    --model_name="my_noncontextual_model" \
    --fc_hidden_size="512" \
    --dropout="0.0" \
    --learning_rate="0.001" \
    --dataset="coref149" \
    --embedding_size="100" \
    --use_pretrained_embs="word2vec" \
    --freeze_pretrained \
    --fixed_split
```

### Contextual model (ELMo)

Contextual model with ELMo embeddings

```bash
$ python contextual_model_elmo.py \
    --model_name="my_elmo_model" \
    --dataset="coref149" \
    --fc_hidden_size="64" \
    --dropout="0.4" \
    --learning_rate="0.005" \
    --num_epochs="20" \
    --freeze_pretrained \
    --fixed_split
```

### Contextual model (BERT)

Contextual model with BERT embeddings

```bash
$ python contextual_model_bert.py \
    --model_name="my_bert_model" \
    --dataset="coref149" \
    --fc_hidden_size="64" \
    --dropout="0.4" \
    --learning_rate="0.001" \
    --num_epochs="20" \
    --freeze_pretrained \
    --fixed_split
```

# Docker setup

Rest API is is provided by FastAPI/uvicorn.

## Installation

Install base and API requirements by running 
```bash
$ pip install -r requirements.txt
$ pip install -r requirements-api.txt
```

## Setup & running

Running REST API requires two environment variables:
- `COREF_MODEL_PATH`, containing the path to the coref model
- `CLASSLA_RESOURCES_DIR`, a path to where the CLASSLA resources will be downloaded (if are not already) and used


REST API can then be ran by moving into the `src` directory and running the `uvicorn` module:

```sh
$ cd ./src
$ python -m uvicorn rest_api:app --port=5020
```

You can, of course, define required env variables on the fly when running, for example:

```sh
$ CLASSLA_RESOURCES_DIR=.\classla_resources \
  COREF_MODEL_PATH=.\contextual_model_bert \
  python -m uvicorn rest_api:app --port=5020
```

Assuming everything went smoothly, the API will become available at http://localhost:5020/predict/coref.
```
...
INFO:     Uvicorn running on http://127.0.0.1:5020 (Press CTRL+C to quit)
```

To test it, try sending a request with **curl**:
```sh
$ curl -X POST -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"threshold": 0.60, "return_singletons": true, "text": "Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer."}' \
  "http://localhost:5020/predict/coref"
```

### Documentation

After starting up the API, the OpenAPI/Swagger documentation will also become accessible at http://localhost:5020/docs.

## Building a Docker image

To build the docker image, run 

```sh
docker build --tag slo-coref -f Dockerfile .
```

### Running a Docker container

To run the docker image, run the following command with properly fixed mount `source` paths:

```sh
docker run --rm -it --name slo-coref \
  -p 5020:5020 \
  --env COREF_MODEL_PATH="/app/data/bert_based/" \
  --mount type=bind,source="/path/to/contextual_model_bert/",destination="/app/data/bert_based/",ro \
  slo-coref
```

**Note:** If container is killed at startup, increase Docker memory limits (8GB is proposed).

---

> Operacijo Razvoj slovenščine v digitalnem okolju sofinancirata Republika Slovenija in Evropska unija iz Evropskega sklada za regionalni razvoj. Operacija se izvaja v okviru Operativnega programa za izvajanje evropske kohezijske politike v obdobju 2014-2020.

![](Logo_EKP_sklad_za_regionalni_razvoj_SLO_slogan.jpg)

----

SloCOREF tool (modeling and training) was partly financed by [CLARIN.SI](https://www.clarin.si).

<center>
	<img src="clarin-logo.png" width="500pt" />
<center>
