import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class SgnetFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(SgnetFeatureExtractor, self).__init__()
        self.embbed_size = args.output_dim
        self.box_embed = nn.Sequential(nn.Linear(args.input_dim, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)

        return embedded_box_input
    

class DescFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(DescFeatureExtractor, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = SgnetFeatureExtractor(args)
        
    def forward(self, desc, frame_len):
        inputs = self.tokenizer(desc, return_tensors='pt', padding=True, truncation=True).to(device)
        h = self.model(**inputs)
        embeddings = h.last_hidden_state.mean(dim=1) # (bs, 768)
        embeddings = embeddings.unsqueeze(1).repeat(1, frame_len, 1) # (bs, 15, 768)
        outputs = self.fc(embeddings)
        
        return outputs
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False