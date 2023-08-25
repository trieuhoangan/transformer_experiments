from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils
import numpy as np


from dataclasses import dataclass
import torch.nn.functional as F
from fairseq import metrics, utils

from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@register_criterion('join_label_smoothed_cross_entropy')
class JoinLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    # @classmethod
    # def build_criterion(cls, args, task,sentence_avg,label_smoothing,mask_loss_weight):
    #     """Construct a criterion from command-line args."""
    #     return cls(args, task,sentence_avg,label_smoothing,mask_loss_weight)
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        mask_loss_weight,
        parse_penalty,
        encoder_lisa_layer,
        encoder_layers,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.mask_loss_weight=mask_loss_weight
        self.parse_penalty = parse_penalty
        self.lisa_layer = encoder_lisa_layer
        self.encoder_layers = encoder_layers
        self.src_tags_dict = task.source_tags_dictionary

        self.map_dictionary = dict()
        for k, v in self.src_tags_dict.indices.items():
            try:
                self.map_dictionary[v] = float(k)
            except:
                pass

    def add_args(parser):
        parser.add_argument('--mask-loss-weight', default=0.2, type=float,
                            help='weight of mask loss')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # parser.add_argument('--encoder-layers', type=int, metavar='N',
        #                     help='num encoder layers')
        parser.add_argument('--parse-penalty', default=1.0, type=float,
                            help='penalty of parsing loss')    
        # parser.add_argument('--encoder-lisa-layer', default=None, type=int,
        #                     help='LISA layer')
    

    def forward(self, model, sample, reduce=True,show=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

            # return super().forward(model, sample, reduce=reduce)
        net_output, net_output_mask = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        src_tags = sample['net_input']['src_tags']
        l = src_tags.size(1) - 1
        heads = torch.cuda.LongTensor(np.vectorize(lambda e: self.map_dictionary.get(e, l))(src_tags.cpu()))
        
        dep_targets = heads
        dep_probabilities = net_output[2]
        attn_loss = F.cross_entropy(dep_probabilities, dep_targets, reduction='sum')
        # Combine losses
        multi_loss = loss + self.parse_penalty * attn_loss
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        src_len=net_output[1]["mask"][0].size()[-1]
        mask_ave = net_output[1]["mask"][0].mean(dim=0).mean(dim=0).mean(dim=-1).sum()
        gate_ave=net_output[1]["gate"][0].mean(dim=0).mean(dim=0).sum()
        
        mask_loss, _ = self.compute_loss(model, net_output_mask, sample, reduce=reduce)
        p_norm = torch.norm(1-net_output[1]["mask"][0], p=2)/src_len
        mask_loss_final = -mask_loss+self.mask_loss_weight*p_norm
        
        logging_output = {
            "loss": loss.data,
            "mask_loss": mask_loss.data,
            "p2":p_norm.data,
            "nll_loss": nll_loss.data,
            'mask_ave': mask_ave.data,
            "gate_ave": gate_ave.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            'multi_loss': utils.item(multi_loss.data) if reduce else multi_loss.data,
            'attn_loss': utils.item(attn_loss.data) if reduce else attn_loss.data
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        del mask_ave, nll_loss,mask_loss,p_norm
        # print("got in right criterion")
        return multi_loss, mask_loss_final, sample_size, logging_output


    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)
        # print(logging_outputs)
        mask_loss_sum = sum(log.get('mask_loss', 0) for log in logging_outputs)
        # mask_loss_final_sum = sum(log.get('mask_loss_final', 0) for log in logging_outputs)
        p_sum = sum(log.get('p2', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        mask_sum = sum(log.get('mask_ave', 0) for log in logging_outputs) 
        gate_sum=sum(log.get('gate_ave', 0) for log in logging_outputs)


        metrics.log_scalar('mask_loss', mask_loss_sum / sample_size / math.log(2), sample_size, round=6)
        # metrics.log_scalar('mask_loss_final', mask_loss_final_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('p_2', p_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('mask_ave', mask_sum / sample_size, sample_size, round=3)
        metrics.log_scalar('gate_ave', gate_sum / sample_size, sample_size, round=3)