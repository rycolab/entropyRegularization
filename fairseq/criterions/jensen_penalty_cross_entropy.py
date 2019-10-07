import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch.distributions as d

def smoothed_nll_loss(lprobs, target, alpha, beta, dist, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    ent_p = -torch.exp(lprobs).mul(lprobs).sum(dim=-1, keepdim=True)
    uni_probs = torch.ones(lprobs.size(), device=lprobs.device) * dist
    print(uni_probs.sum(dim=1).isclose(torch.ones(lprobs.size(0), device=lprobs.device)))
    assert uni_probs.sum(dim=1).isclose(torch.ones(lprobs.size(0), device=lprobs.device)).all()

    comb = torch.exp(lprobs) * alpha + uni_probs * (1. - alpha)
    ent_c = -torch.log(comb).mul(comb).sum(dim=-1, keepdim=True)
    ent = ent_c - alpha * ent_p # not including (1 - alpha) * ent_u since it's constant

    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        ent = ent[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        ent = ent.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        ent = ent.sum()
    loss = nll_loss - beta * ent
    return loss, nll_loss


@register_criterion('jensen_cross_entropy')
class JensonCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.alpha
        self.beta = args.beta
        tgt_dict = task.target_dictionary
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
        parser.add_argument('--T', default=1., type=float, metavar='N',
                            help='unigram distribution annealing factor')
        parser.add_argument('--beta', default=0., type=float, metavar='D',
                            help='weight of penalty')
        parser.add_argument('--alpha', default=0.5, type=float, metavar='D',
                            help='alpha parameter for divergence')
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
            lprobs, target, self.alpha, self.beta, self.unigram_dist, ignore_index=self.padding_idx, reduce=reduce,
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
