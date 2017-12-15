import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CNNNCE(nn.Module):
    def __init__(self, input_embedding, cnn, sample_embedding, noise_scaling):
        super(CNNNCE, self).__init__()

        self.input_embedding_ = input_embedding
        self.sample_embedding_ = sample_embedding
        self.cnn_ = cnn
        self.noise_scaling_ = nn.Parameter(
            torch.log(torch.FloatTensor([noise_scaling])))

    def parameters(self):
        for param in self.input_embedding.parameters():
            yield param
        for param in self.sample_embedding.parameters():
            yield param
        for param in self.cnn.parameters():
            yield param

    @property
    def noise_scaling(self):
        return self.noise_scaling_

    @property
    def input_embedding(self):
        return self.input_embedding_

    @property
    def sample_embedding(self):
        return self.sample_embedding_

    @property
    def cnn(self):
        return self.cnn_

    def forward(self, inputs):
        batch_size = inputs.samples.size(0)
        embedded_targets = self.sample_embedding(inputs.samples)
        embedded_seq = self.input_embedding(inputs.sequence)
        feature_map = self.cnn.encoder_state_output(embedded_seq)
        score = torch.bmm(
            embedded_targets, feature_map.view(batch_size, -1, 1)).view(
            batch_size, -1)
        noise_term = torch.log(inputs.noise_probs).add_(self.noise_scaling)
        noise_term.masked_fill_(inputs.samples.eq(0), 0)
        logit = score.sub_(noise_term)
        return logit

    def nearest_neighbors(self, inputs, k):
        target_vocab_size = self.sample_embedding.vocab_size

        embedded_seq = self.input_embedding(inputs.sequence)
        feature_map = self.cnn.encoder_state_output(embedded_seq)
        targets = Variable(inputs.sequence.data.new(
            [[i for i in range(target_vocab_size + 1)]]))
        embedded_targets = self.sample_embedding(targets).view(
            target_vocab_size + 1, -1)
        scores = feature_map.mm(embedded_targets.transpose(1,0))
        scores_sorted, indices_sorted = torch.sort(scores, 1, descending=True)

        return indices_sorted[:,:k], scores_sorted[:,:k] 

    @property
    def input_module(self):
        return self.input_module_

    @property
    def encoder_module(self):
        return self.encoder_module_

    @property
    def predictor_module(self):
        return self.predictor_module_
