from faker import Faker
import json, sys, os, random
import nltk

tok = nltk.TweetTokenizer()

# we use French localization to keep the same French language models we used in the paper for the real confidential datasets
fake = Faker('fr_FR')

# split sizes
train_size = 1000
val_size = 250
test_size = 250
unlabeled_size = 4300

dummy_id = 0

def generate_random_entry(split):
    # authors refers to speaker role
    entry = {"id": None, "status": None, "satisfaction": None, "texts": None, "labels": None, "authors": None, "split": split}

    global dummy_id
    entry["id"] = dummy_id
    dummy_id+=1

    # random conversation level labels
    entry["status"] = random.randint(0, 4)
    entry["satisfaction"] = random.randint(0, 6)
    
    # random conversation messages with random length in range
    texts = []
    # additional utterance level label that we have in the original dataset but did not use in the paper
    labels = []
    speaker_roles = []
    for i in range(random.randint(5, 15)):
        labels.append(random.randint(0,9))
        speaker_roles.append(random.randint(2,3))
        tokenized = tok.tokenize(fake.text().replace("\n",""))
        texts.append(" ".join(tokenized))

    entry['labels'] = labels
    entry['authors'] = speaker_roles
    entry["texts"] = texts

    return json.dumps(entry)


trainset = [ generate_random_entry("train") for i in range(train_size)]
valset = [ generate_random_entry("val") for i in range(val_size)]
testset = [ generate_random_entry("test") for i in range(test_size)]
unlabeledset = [ generate_random_entry("unlabeled") for i in range(unlabeled_size)]

with open('dummy_dataset.json', 'w') as f:
    f.writelines("\n".join(trainset+valset+testset+unlabeledset))