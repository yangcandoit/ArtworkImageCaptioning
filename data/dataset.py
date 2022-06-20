import os
import numpy as np
import itertools
import collections
import torch
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO
import json
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, examples, fields,config,mode,):
        self.examples = examples
        self.fields = dict(fields)
        self.config=config
        self.mode=mode

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)
            # print('111',tensor.shape)
            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        
        example = self.examples[i]
        # print(dir(example))
        data = []
        for field_name, field in self.fields.items():
            if field_name=='image':           
                print(getattr(example, field_name))     
                data.append(field.preprocess(getattr(example, field_name),self.mode,self.config))
            else:
                data.append(field.preprocess(getattr(example, field_name)))
        # print(data)
        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(MyDataset):
    def __init__(self, examples, fields, dictionary,config,mode):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields,config,mode)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
            
            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(MyDataset):
    def __init__(self, examples, fields, key_fields,config,mode):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = MyDataset(key_examples, key_fields,config,mode)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary,config,mode)
        super(DictionaryDataset, self).__init__(examples, fields,config,mode)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(MyDataset):
    def __init__(self, examples, fields,config,mode):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields,config,mode)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image',config=self.config,mode=self.mode)
        return dataset

    @property
    def splits(self):
        raise NotImplementedError

class ICON(PairedDataset):
    def __init__(self, image_field, text_field, root=None,config=None,mode=''):
        
        self.root=root
        self.cap='tdata.json'
        
        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples()
        
        examples = self.train_examples + self.val_examples + self.test_examples
        super(ICON, self).__init__(examples, {'image': image_field, 'text': text_field},config,'')

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields,self.config,mode='train')
        val_split = PairedDataset(self.val_examples, self.fields,self.config,mode='val')
        test_split = PairedDataset(self.test_examples, self.fields,self.config,mode='test')
        return train_split, val_split, test_split

    # @classmethod
    def get_samples(self):
        with open(os.path.join(self.root, self.cap),'r') as load_f:
            captions = json.load(load_f)
        self.samples=[]
        for i in captions:

            example = Example.fromdict({'image': os.path.join(self.root, i), 'text': captions[i]})
            self.samples.append(example)

        train_samples=self.samples[:int(len(self.samples)*0.8)]
        val_samples=self.samples[int(len(self.samples)*0.8):int(len(self.samples)*0.9)]
        test_samples=self.samples[int(len(self.samples)*0.9):]

        return train_samples, val_samples, test_samples

