import torch
from torch.autograd import Variable
from collections import namedtuple


class ContextSampler(object):
    def __init__(self, dataset, frequencies, num_positive=5, 
                 num_negative=10, pad_value=0, pos_inv_freq=False):

        self.dataset_ = dataset
        self.frequencies_ = frequencies
        self.distribution_ = frequencies / frequencies.sum()
        self.num_positive_ = num_positive
        self.num_negative_ = num_negative
        self.pad_value_ = pad_value
        self.pos_inv_freq_ = pos_inv_freq
        self.inputs_wrapper_ = namedtuple(
            "inputs", ["sequence", "length", "samples", "noise_probs"])
        self.batch_wrapper_ = namedtuple("Batch", ["inputs", "targets"])

    @property
    def inputs_wrapper(self):
        return self.inputs_wrapper_

    @property
    def batch_wrapper(self):
        return self.batch_wrapper_

    @property
    def pos_inv_freq(self):
        return self.pos_inv_freq_

    @property
    def pad_value(self):
        return self.pad_value_

    @property
    def size(self):
        return self.dataset.size

    @property
    def batch_size(self):
        return self.dataset.batch_size

    @property
    def dataset(self):
        return self.dataset_

    @property
    def frequencies(self):
        return self.frequencies_

    @property
    def distribution(self):
        return self.distribution_

    @property
    def num_positive(self):
        return self.num_positive_

    @property
    def num_negative(self):
        return self.num_negative_

    def get_noise_probs(self, indices):
        noise_probs_flat = self.distribution.index_select(0, indices.view(-1))
        noise_probs = noise_probs_flat.view(indices.size())
        return noise_probs

    def draw_positive_samples(self, inputs):
        freqs = self.make_positive_distribution(inputs)
        return torch.multinomial(freqs, self.num_positive, replacement=False)

    
    def set_gpu(self):
        if self.dataset.gpu > -1:
            self.distribution_ = self.distribution_.cuda(self.dataset.gpu)
            self.frequencies_ = self.frequencies_.cuda(self.dataset.gpu)
        else:
            self.distribution_ = self.distribution_.cpu()
            self.frequencies_ = self.frequencies_.cpu()



    def sample_partial_sequence(self, batch):
        pos_dist = self.get_positive_distribution(
            batch.inputs.sequence, self.pos_inv_freq)
        pos_samples = self.draw_context_sample(
            batch.inputs.sequence, pos_dist, self.num_positive)
        neg_samples = self.draw_negative_sample(
            batch.inputs.sequence.size(0), self.num_positive, 
            self.num_negative)
        samples = Variable(torch.cat([pos_samples, neg_samples], 2))
        targets = self.distribution.new().resize_(samples.size()).fill_(0)
        targets[:,:,0] = 1
        targets = Variable(targets)

        noise_probs = self.distribution.index_select(0, samples.data.view(-1))
        noise_probs = Variable(noise_probs.view(samples.size()))

        inputs = self.inputs_wrapper(
            batch.inputs.sequence, batch.inputs.length, samples, noise_probs)
        batch_sample = self.batch_wrapper(
            inputs, targets)
        return batch_sample
    
    def get_positive_distribution(self, inputs, inv_freq):
        input_flat = inputs.data.view(-1)
        freqs = self.frequencies.index_select(
            0, inputs.data.view(-1)).view(inputs.size())
        if inv_freq:
            mask = inputs.data.eq(0)
            return (1 / freqs).masked_fill_(mask, 0)
        else:
            return freqs

    def draw_context_sample(self, context, dist, num_samples):
        sample_idx = torch.multinomial(
            dist, num_samples, replacement=False)
        sample = torch.gather(
            context.data.unsqueeze(2), 1, sample_idx.unsqueeze(2))
        return sample

    def draw_negative_sample(self, batch_size, num_positive, num_negative):
        samples = torch.multinomial(
            self.distribution, 
            batch_size * num_positive * num_negative, 
            replacement=True)
        return samples.view(batch_size, num_positive, num_negative)




    def sample_complete_sequence(self, batch):
        pass

    def iter_batch(self):
        self.set_gpu()

        for batch in self.dataset.iter_batch():
            if self.num_positive > 0:
                yield self.sample_partial_sequence(batch)
            else:
                raise NotImplementedError()
#                yield self.sample_complete_sequence(batch)
#            continue
#            
#            batch_size = batch.inputs.sequence.size(0)
#            input_size = batch.inputs.sequence.size(1)
#
#            mask = batch.inputs.sequence.data.eq(self.pad_value)
#            mask_3d = mask.view(
#                batch_size, -1, 1).repeat(1, 1, self.num_negative + 1)
#
#            neg_samples = self.draw_negative_samples(batch_size, input_size)
#            if self.dataset.gpu > -1:
#                neg_samples = neg_samples.cuda(self.dataset.gpu)
#
#            neg_samples = Variable(
#                    neg_samples.masked_fill_(mask_3d[:,:,:-1], self.pad_value))
#
#            pos_samples = batch.inputs.sequence.view(batch_size, input_size, 1)
#            samples = torch.cat([pos_samples, neg_samples], 2).view(
#                batch_size, -1)
#            
#            labels = torch.zeros(batch_size, input_size, self.num_negative + 1)
#            labels[:,:,0].fill_(1)
#            if self.dataset.gpu > -1:
#                labels = labels.cuda(self.dataset.gpu)
#            labels.masked_fill_(mask_3d, -1)
#            labels = Variable(labels.view(batch_size, -1))
#            
#            noise_probs = Variable(self.get_noise_probs(samples.data))
#
#            inputs = namedtuple(
#                "Inputs", ["sequence", "length", "samples", "noise_probs"])(
#                    batch.inputs.sequence, batch.inputs.length, samples,
#                    noise_probs)
#            wrapped_batch = namedtuple("Batch", ["inputs", "targets"])(
#                inputs, labels)
#            yield wrapped_batch
