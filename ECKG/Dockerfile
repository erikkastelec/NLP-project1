FROM python:3.9

WORKDIR /app

ENV CLASSLA_RESOURCES_DIR '/app/data/classla'

COPY ./src /app
COPY ./requirements.txt /app/
COPY ./requirements-api.txt /app/

# For some reason, two versions of torch get installed when installing requirements?
# Installing specific torch version before installing the rest solves this problem.
RUN pip install torch==1.10.0+cpu \
    torchvision==0.11.1+cpu \
    torchaudio==0.10.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install -r requirements.txt

# Install additional requirements to run REST API
RUN pip install -r requirements-api.txt

RUN python -c "import classla; import os; \
               CLASSLA_RESOURCES_DIR = os.getenv('CLASSLA_RESOURCES_DIR', None); \
               processors = 'tokenize,pos,lemma,ner'; \
               classla.download('sl', processors=processors);"

EXPOSE 5020

CMD ["uvicorn", "rest_api:app", "--host", "0.0.0.0", "--port", "5020"]