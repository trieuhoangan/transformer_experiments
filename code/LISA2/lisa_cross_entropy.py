# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import math
import numpy as np

import torch
from dataclasses import dataclass
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import LegacyFairseqCriterion, FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class LisaCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("params.optimization.sentence_avg")

@register_criterion('lisa_cross_entropy', dataclass=LisaCrossEntropyCriterionConfig)
class LISACrossEntropyCriterion(LegacyFairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.parse_penalty = args.parse_penalty
        self.lisa_layer = args.encoder_lisa_layer
        self.encoder_layers = args.encoder_layers
        self.src_tags_dict = task.source_tags_dictionary

        self.map_dictionary = dict()
        for k, v in self.src_tags_dict.indices.items():
            try:
                self.map_dictionary[v] = float(k)
            except:
                pass
 
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--parse-penalty', default=1.0, type=float,
                            help='penalty of parsing loss')
        # fmt: on
    
    def forward(self, model, sample, reduce=True):
        
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # Compute distance function
        src_tags = sample['net_input']['src_tags']
        l = src_tags.size(1) - 1
        heads = torch.cuda.LongTensor(np.vectorize(lambda e: self.map_dictionary.get(e, l))(src_tags.cpu()))
        
        dep_targets = heads
        
        net_output = model(**sample['net_input'])
        # Compute label smoothed cross entropy (NMT loss)
        loss, nll_loss = self.compute_xent_loss(model, net_output, sample, reduce=reduce)
        # Compute parsing cross entropy (LISA loss)
        dep_probabilities = net_output[2]
        origin_depprob = dep_probabilities
        try:
            attn_loss = F.cross_entropy(dep_probabilities, dep_targets, reduction='sum')
            # Combine losses
            multi_loss = loss + self.parse_penalty * attn_loss


            sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'multi_loss': utils.item(multi_loss.data) if reduce else multi_loss.data,
                'attn_loss': utils.item(attn_loss.data) if reduce else attn_loss.data,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return multi_loss, sample_size, logging_output
        except:
            print("sample ",sample['net_input']['src_tokens'].shape)
            print("in put ",dep_targets.shape)
            # print("out put ",dep_probabilities.shape)
            # print("origin out put ",origin_depprob.shape)
    def compute_xent_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'multi_loss': sum(log.get('multi_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'attn_loss': sum(log.get('attn_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
    @classmethod
    def reduce_metrics(cls,logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

