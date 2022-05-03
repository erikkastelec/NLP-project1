from ECKG.src.baseline import BaselineController, MentionPairFeatures
from contextual_model_bert import ContextualControllerBERT
from data import read_classla, read_corpus
from eventify import *
from helper_functions import *

# coreferece or events
MODE = "coreference"
# sl or en
LANGUAGE = "sl"

if __name__ == '__main__':
    if MODE == "coreference":

        model_name = "baseline_model_senticoref"
        learning_rate = 0.05
        dataset_name = "senticoref"
        # BASELINE
        model = BaselineController(MentionPairFeatures.num_features(),
                                   model_name=model_name,
                                   learning_rate=learning_rate,
                                   dataset_name=dataset_name
                                   )
        model = BaselineController.from_pretrained("baseline_model/baseline_model_senticoref")

        # CONTEXTUAL BERT
        model = ContextualControllerBERT.from_pretrained("contextual_model_bert/fold7_0")

        # read corpus
        documents = read_corpus(dataset_name)
        # test on first document
        eval_doc = model._train_doc(documents[0], eval_mode=True)
        # mapped_cluster_dict = {}
        # for key, values in documents[0].mapped_clusters.items():
        #     if values is None:
        #         values = "None"
        #     try:
        #         mapped_cluster_dict["values"].append(key)
        #     except Exception:
        #         mapped_cluster_dict["values"] = key
        for cluster in documents[0].clusters:
            print([" ".join([y.raw_text for y in documents[0].mentions[x].tokens]) for x in cluster])
        for key, values in eval_doc[0].items():
            if key is None:
                key = "None"
            print(" ".join([y.raw_text for y in documents[0].mentions[key].tokens]), ":")
            print([" ".join([y.raw_text for y in documents[0].mentions[x].tokens]) for x in values])
        # Evaluation for conll format produced by classla
        classla.download('sl')
        pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
        text = "Ljubljana - Evropska komisija je Sloveniji za letos napovedala 1,9 - odstotno rast , za naslednje leto pa 2,5 - odstotno , kar je nad povprečjem v območju evra in EU . Sicer pa bo Slovenija letos po napovedih iz Bruslja beležila 5,8 - odstotni javnofinančni primanjkljaj , naslednje leto pa petodstotnega , kar je slabše od povprečja v območju evra in EU . Javni dolg Slovenije se bo po napovedih komisije vztrajno večal , a bo še vedno bistveno pod povprečjem območja evra in EU ter bistveno pod referenčno mejo 60 odstotkov bruto domačega proizvoda ( BDP ) , ki jo določa pakt za stabilnost in rast . Za letos komisija Sloveniji napoveduje 42,8 - odstotni dolg , za prihodnje leto pa 46 - odstotnega . Poleg tega Evropska komisija Sloveniji za letos napoveduje 2,6 - odstotno inflacijo , za prihodnje leto pa 2,1 - odstotno . Inflacija v Sloveniji bo tako letos po napovedih komisije enaka kot v povprečju v območju evra in pod povprečjem EU , prihodnje leto pa nekoliko nad povprečjem v območju evra in celotne EU . Po napovedih iz Bruslja bo Slovenija letos beležila večjo brezposelnost kot lani . Medtem ko je bila ta lani po ocenah komisije 7,3 - odstotna , naj bi bila letos 8,2 - odstotna , prihodnje leto pa osemodstotna . To je še vedno manj kot v povprečju v območju evra in celotne EU , kjer naj bi bila letos 10 - odstotna , prihodnje leto pa 9,7 - odstotna oziroma 9,5 - odstotna in 9,1 - odstotna . V lanski novembrski gospodarski napovedi je komisija Sloveniji za letos prav tako napovedala 1,9 - odstotno rast , za leto 2012 pa 2,6 - odstotno , a njene napovedi za javnofinančni primanjkljaj Slovenije so bile novembra boljše kot danes . Novembra je namreč napovedala , da bo ta v Sloveniji letos 5,3 - odstoten , naslednje leto pa 4,7 - odstoten . Slovenija je sicer - kot velika večina držav EU - zaradi prekomernega javnofinančnega primanjkljaja v postopku Evropske komisije . Evropska komisija je kot rok za znižanje javnofinančnega primanjkljaja Slovenije pod tri odstotke BDP , kar kot zgornjo mejo določa pakt za stabilnost in rast , določila leto 2013 . Slovenija je po navedbah finančnega ministrstva v programu stabilnosti , ki ga je posredovala Evropski komisiji , zapisala , da za leto 2012 načrtuje znižanje primanjkljaja na 3,9 odstotka BDP . Na zvišanje letošnjega primanjkljaja bo sicer po navedbah komisije vplivala dokapitalizacija največje slovenske banke , Nove Ljubljanske banke . Glede primanjkljaja v letu 2012 je komisija zapisala , da je izračunan ob predpostavki , da ne bo dodatnih ukrepov . Glede nadaljnjih ukrepov za zajezitev izdatkov v letu 2012 sicer komisija ugotavlja , da ti v glavnem še niso podrobno opredeljeni in dogovorjeni . Glede gospodarske rasti pa Evropska komisija v primeru Slovenije ugotavlja razmeroma medlo okrevanje , ki temelji na izvozu . Ta pa je po njenih navedbah ohromljen zaradi izgub konkurenčnosti , ki so se nabrale pred in med krizo , ter zaradi usmerjenosti izvoza na gospodarsko šibek Zahodni Balkan . Drugi dejavnik , ki zavira gospodarsko rast Slovenije , so po navedbah Evropske komisije razmere v gradbeništvu . Poleg tega se je po navedbah komisije v prizadevanjih bank za izboljšanje svojih bilanc skrčil tudi tok posojil v realno gospodarstvo"
        # Analyze using classla/stanza to get Document format
        data = pipeline(text)
        # Convert to conll format
        data = data.to_conll()
        # Convert to format used by slovene-coreference-resolution library (https://github.com/matejklemen/slovene-coreference-resolution)
        # TODO: need to fix some mistakes when converting formats
        data = read_classla(data)
        # res, _ = model._train_doc(data3, eval_mode=True)
    else:
        if LANGUAGE == "sl":
            # Initialize Slovene classla pipeline
            classla.download('sl')
            pipeline = classla.Pipeline("sl", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
            text = "Ljubljana - Evropska komisija je Sloveniji za letos napovedala 1,9 - odstotno rast , za naslednje leto pa 2,5 - odstotno , kar je nad povprečjem v območju evra in EU . Sicer pa bo Slovenija letos po napovedih iz Bruslja beležila 5,8 - odstotni javnofinančni primanjkljaj , naslednje leto pa petodstotnega , kar je slabše od povprečja v območju evra in EU . Javni dolg Slovenije se bo po napovedih komisije vztrajno večal , a bo še vedno bistveno pod povprečjem območja evra in EU ter bistveno pod referenčno mejo 60 odstotkov bruto domačega proizvoda ( BDP ) , ki jo določa pakt za stabilnost in rast . Za letos komisija Sloveniji napoveduje 42,8 - odstotni dolg , za prihodnje leto pa 46 - odstotnega . Poleg tega Evropska komisija Sloveniji za letos napoveduje 2,6 - odstotno inflacijo , za prihodnje leto pa 2,1 - odstotno . Inflacija v Sloveniji bo tako letos po napovedih komisije enaka kot v povprečju v območju evra in pod povprečjem EU , prihodnje leto pa nekoliko nad povprečjem v območju evra in celotne EU . Po napovedih iz Bruslja bo Slovenija letos beležila večjo brezposelnost kot lani . Medtem ko je bila ta lani po ocenah komisije 7,3 - odstotna , naj bi bila letos 8,2 - odstotna , prihodnje leto pa osemodstotna . To je še vedno manj kot v povprečju v območju evra in celotne EU , kjer naj bi bila letos 10 - odstotna , prihodnje leto pa 9,7 - odstotna oziroma 9,5 - odstotna in 9,1 - odstotna . V lanski novembrski gospodarski napovedi je komisija Sloveniji za letos prav tako napovedala 1,9 - odstotno rast , za leto 2012 pa 2,6 - odstotno , a njene napovedi za javnofinančni primanjkljaj Slovenije so bile novembra boljše kot danes . Novembra je namreč napovedala , da bo ta v Sloveniji letos 5,3 - odstoten , naslednje leto pa 4,7 - odstoten . Slovenija je sicer - kot velika večina držav EU - zaradi prekomernega javnofinančnega primanjkljaja v postopku Evropske komisije . Evropska komisija je kot rok za znižanje javnofinančnega primanjkljaja Slovenije pod tri odstotke BDP , kar kot zgornjo mejo določa pakt za stabilnost in rast , določila leto 2013 . Slovenija je po navedbah finančnega ministrstva v programu stabilnosti , ki ga je posredovala Evropski komisiji , zapisala , da za leto 2012 načrtuje znižanje primanjkljaja na 3,9 odstotka BDP . Na zvišanje letošnjega primanjkljaja bo sicer po navedbah komisije vplivala dokapitalizacija največje slovenske banke , Nove Ljubljanske banke . Glede primanjkljaja v letu 2012 je komisija zapisala , da je izračunan ob predpostavki , da ne bo dodatnih ukrepov . Glede nadaljnjih ukrepov za zajezitev izdatkov v letu 2012 sicer komisija ugotavlja , da ti v glavnem še niso podrobno opredeljeni in dogovorjeni . Glede gospodarske rasti pa Evropska komisija v primeru Slovenije ugotavlja razmeroma medlo okrevanje , ki temelji na izvozu . Ta pa je po njenih navedbah ohromljen zaradi izgub konkurenčnosti , ki so se nabrale pred in med krizo , ter zaradi usmerjenosti izvoza na gospodarsko šibek Zahodni Balkan . Drugi dejavnik , ki zavira gospodarsko rast Slovenije , so po navedbah Evropske komisije razmere v gradbeništvu . Poleg tega se je po navedbah komisije v prizadevanjih bank za izboljšanje svojih bilanc skrčil tudi tok posojil v realno gospodarstvo"

        else:
            # Initialize English stanza pipeline
            stanza.download("en")
            pipeline = stanza.Pipeline("en", processors='tokenize,ner, lemma, pos, depparse', use_gpu=True)
            text = "Russian forces are pressing an offensive in the direction of Sloviansk, an important town in the Donetsk region, according to Ukraine's military, while Russia’s Belgorod region, which borders Ukraine, was hit by two explosions early Monday, the local governor said."

        # Analyze using classla/stanza to get Document format
        data = pipeline(text)
        # Run named entity deduplication/resolution
        deduplication_mapper = deduplicate_named_entities(data)
        print(deduplication_mapper)
        x = [" ".join([y.text for y in x.tokens]) for x in data.entities]
        print(x)
        e = Eventify(language=LANGUAGE)
        # Extract SVO triplets / events
        events = e.eventify(text)
        print(events)
