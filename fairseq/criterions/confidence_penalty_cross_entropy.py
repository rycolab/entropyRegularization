# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch.distributions as d

def smoothed_nll_loss(lprobs, target, beta, ignore_index=None, reduce=True, dist=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if dist is not None:
        uni_probs = torch.ones(lprobs.size(), device=lprobs.device) * dist
        divergence = (torch.exp(lprobs).mul(lprobs) - torch.exp(lprobs).mul(torch.log(uni_probs))).sum(dim=-1, keepdim=True)
        assert divergence.sum() >= 0
    else:
        divergence = torch.exp(lprobs).mul(lprobs).sum(dim=-1, keepdim=True)
    
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        divergence = divergence[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        divergence = divergence.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        divergence = divergence.sum()
    loss = nll_loss + beta * divergence
    return loss, nll_loss


@register_criterion('confidence_penalty_cross_entropy')
class ConfidencePenaltyCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.beta
        tgt_dict = task.target_dictionary
        if args.T >= 1:
            self.use_uni_dist = True
            self.unigram_dist = torch.Tensor(tgt_dict.count)
            self.unigram_dist[tgt_dict.eos_index] = self.unigram_dist[tgt_dict.index('.')]
            self.unigram_dist = self.unigram_dist.pow(1./args.T)
            self.unigram_dist = self.unigram_dist/self.unigram_dist.sum()
            if torch.cuda.is_available() and not args.cpu:
                self.unigram_dist = self.unigram_dist.cuda()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--beta', default=0., type=float, metavar='D',
                            help='weight of penalty')
        parser.add_argument('--T', default=0, type=float, metavar='N',
                            help='unigram distribution annealing factor')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            dist=self.unigram_dist if self.use_uni_dist else None
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
