import json
import scispacy
import spacy
from spacy.tokens import Doc
from more_itertools import locate


#!pip install scispacy
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz

nlp = spacy.load('en_core_sci_sm')


def custom_tokenizer(text):
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)

nlp.tokenizer = custom_tokenizer

class JsonInputAugmenter():
    def __init__(self):
        basepath = './data/datasets/scierc/'
        self.input_dataset_paths  = [basepath + 'scierc_train.json', basepath + 'scierc_dev.json', basepath + 'scierc_train_dev.json',     basepath + 'scierc_test.json']
        self.output_dataset_paths = [basepath + 'scierc_train_aug.json', basepath + 'scierc_dev_aug.json', basepath + 'scierc_train_dev_aug.json', basepath + 'scierc_test_aug.json'] 

    def augment_docs_in_datasets(self):
        for ipath, opath  in zip(self.input_dataset_paths, self.output_dataset_paths):
            self._augment_docs(ipath, opath)

    def _augment_docs(self, ipath, opath):
        global tokens_dict
        documents = json.load(open(ipath))
        augmented_documents = []
        nmultiroot=0
        for document in documents:
            jtokens = document['tokens']
            jrelations = document['relations']
            jentities = document['entities']
            jorig_id = document['orig_id']

            lower_jtokens = jtokens 
            text = ' '.join(lower_jtokens)

            tokens = nlp(text)            #get annotated tokens
            jtags = [token.tag_ for token in tokens]
            jdeps = [token.dep_ for token in tokens]
            vpos = list(locate(jdeps, lambda x: x == 'ROOT'))
            
            if (len(vpos) != 1):
                flag = 1
                nmultiroot += 1
                print("*** Full sentence:", text)
                for i in vpos:
                    print("ROOT [", i, "]: ", jtokens[i], ", pos tag: ", jtags[i], ", dep: ", jdeps[i])
            else:
                flag = 0

            verb_indicator = [0] * len(jdeps)
            for i in vpos:
                verb_indicator[i] = 1  

            jdep_heads = []
            for i, token in enumerate(tokens):
              if token.head == token:
                 token_idx = 0
              else:
                 token_idx = token.head.i - tokens[0].i + 1
              jdep_heads.append(token_idx)
            if (flag==1):
              print("dep_head: ", jdep_heads)
            d = {"tokens": jtokens, "pos_tags": jtags, "dep_label": jdeps, "verb_indicator": verb_indicator, "dep_head": jdep_heads, "entities": jentities, "relations": jrelations, "orig_id": jorig_id}
            augmented_documents.append(d)
        print("===============  #docs with multiroot = ", nmultiroot)
        with open(opath, "w") as ofile:
            json.dump(augmented_documents, ofile) 

if __name__ == "__main__":
    augmenter = JsonInputAugmenter()
    augmenter.augment_docs_in_datasets()

