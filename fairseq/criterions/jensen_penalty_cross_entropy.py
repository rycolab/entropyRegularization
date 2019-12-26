import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch.distributions as d

def smoothed_nll_loss(lprobs, probs, target, alpha, beta, dist, ignore_index=None, reduce=True, gen="kl"):
    if gen == "kl":
        gen_function = lambda x, y: torch.log(x).mul(x).sum(dim=-1, keepdim=True) if y is None else y.mul(x).sum(dim=-1, keepdim=True)
    elif gen =="eu":
        gen_function = lambda x, y: (x**2).sum(dim=-1, keepdim=True) 
    else:
        exit(1)
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    dist_p = gen_function(probs, lprobs) 
    uni_probs = torch.ones(probs.size(), device=probs.device) * dist
    
    comb = probs * alpha + uni_probs * (1. - alpha)
    dist_c = gen_function(comb, None)
    divergence = 1./((1.-alpha)*alpha)*(-dist_c + alpha * dist_p) # not including + (1-alpha) * dist_u since its constant
    #computationally efficient way of checking divergence property
    assert divergence.sum() >= 0 or divergence.sum() + 1./alpha * gen_function(uni_probs, None).sum() >= 0:
        
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        divergence = divergence[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        divergence = ent.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        divergence = divergence.sum()
    loss = nll_loss + beta * divergence
    return loss, nll_loss


@register_criterion('jensen_cross_entropy')
class JensonCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.alpha = args.alpha
        assert self.alpha < 1.0 and self.alpha > 0.
        self.beta = args.beta
        self.gen = args.generator_function
        tgt_dict = task.target_dictionary
        if args.use_uniform or T < 1.:
            self.dist = torch.ones(len(tgt_dict.symbols)) * 1./len(tgt_dict.symbols)
        else:
            unigram_dist = torch.Tensor(tgt_dict.count)
            #change frequency of eos to frequency of '.' so it's more realistic.
            unigram_dist[tgt_dict.eos_index] = unigram_dist[tgt_dict.index('.')]
            unigram_dist = unigram_dist.pow(1./args.T)
            unigram_dist = unigram_dist/unigram_dist.sum()
            assert not (unigram_dist == 0).any()
            self.dist = unigram_dist
        
        if torch.cuda.is_available() and not args.cpu:
            self.dist = self.dist.cuda()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--T', default=0., type=float, metavar='N',
                            help='unigram distribution annealing factor')
        parser.add_argument('--use-uniform', action='store_true',
                            help='use uniform dist instead')
        parser.add_argument('--beta', default=0., type=float, metavar='D',
                            help='weight of penalty')
        parser.add_argument('--alpha', default=0.5, type=float, metavar='D',
                            help='alpha parameter for divergence; must be < 1 and > 0')
        parser.add_argument('--generator-function', default="kl", choices=['kl', 'eu'],
                            help='divergence generator function')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, entropy = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'entropy': utils.item(entropy.data) if reduce else entropy.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        probs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        probs = probs.view(-1, probs.size(-1))
        entropy = - (lprobs * probs).sum(dim=-1)
        if reduce:
            entropy = entropy.sum()
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = smoothed_nll_loss(
            lprobs, probs, target, self.alpha, self.beta, self.dist, ignore_index=self.padding_idx, 
            reduce=reduce, gen=self.gen)
        return loss, nll_loss, entropy

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'entropy': sum(log.get('entropy', 0) for log in logging_outputs) / ntokens if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
