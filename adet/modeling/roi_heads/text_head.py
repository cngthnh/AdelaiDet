import math
import operator
from typing import Dict, List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY

from adet.layers import conv_with_kaiming_uniform
from ..poolers import TopPooler
from .attn_predictor import ATTPredictor
from dict_trie import Trie	
from editdistance import eval
CTLABELS = [	
    " ",	
    "!",	
    '"',	
    "#",	
    "$",	
    "%",	
    "&",	
    "'",	
    "(",	
    ")",	
    "*",	
    "+",	
    ",",	
    "-",	
    ".",	
    "/",	
    "0",	
    "1",	
    "2",	
    "3",	
    "4",	
    "5",	
    "6",	
    "7",	
    "8",	
    "9",	
    ":",	
    ";",	
    "<",	
    "=",	
    ">",	
    "?",	
    "@",	
    "A",	
    "B",	
    "C",	
    "D",	
    "E",	
    "F",	
    "G",	
    "H",	
    "I",	
    "J",	
    "K",	
    "L",	
    "M",	
    "N",	
    "O",	
    "P",	
    "Q",	
    "R",	
    "S",	
    "T",	
    "U",	
    "V",	
    "W",	
    "X",	
    "Y",	
    "Z",	
    "[",	
    "\\",	
    "]",	
    "^",	
    "_",	
    "`",	
    "a",	
    "b",	
    "c",	
    "d",	
    "e",	
    "f",	
    "g",	
    "h",	
    "i",	
    "j",	
    "k",	
    "l",	
    "m",	
    "n",	
    "o",	
    "p",	
    "q",	
    "r",	
    "s",	
    "t",	
    "u",	
    "v",	
    "w",	
    "x",	
    "y",	
    "z",	
    "{",	
    "|",	
    "}",	
    "~",	
    "ˋ",	
    "ˊ",	
    "﹒",	
    "ˀ",	
    "˜",	
    "ˇ",	
    "ˆ",	
    "˒",	
    "‑",	
]	
def decode(rec):	
    s = ""	
    for c in rec:	
        c = int(c)	
        if c < 104:	
            s += CTLABELS[c]	
        # elif c == 104:	
        #     s += u'口'	
    return s


class SeqConvs(nn.Module):
    def __init__(self, conv_dim, roi_size):
        super().__init__()

        height = roi_size[0]
        downsample_level = math.log2(height) - 2
        assert math.isclose(downsample_level, int(downsample_level))
        downsample_level = int(downsample_level)

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        convs = []
        for i in range(downsample_level):
            convs.append(conv_block(
                conv_dim, conv_dim, 3, stride=(2, 1)))
        convs.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=(4, 1), bias=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class RNNPredictor(nn.Module):
    def __init__(self, cfg):
        super(RNNPredictor, self).__init__()
        # fmt: off
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        conv_dim          = cfg.MODEL.BATEXT.CONV_DIM
        roi_size          = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        # fmt: on

        self.convs = SeqConvs(conv_dim, roi_size)
        self.rnn = nn.LSTM(conv_dim, conv_dim, num_layers=1, bidirectional=True)
        self.clf = nn.Linear(conv_dim * 2, self.voc_size + 1)
        self.recognition_loss_fn = build_recognition_loss_fn()

    def forward(self, x, targets=None):
        # check empty
        if x.size(0) == 0:
            return x.new_zeros((x.size(2), 0, self.voc_size))
        x = self.convs(x).squeeze(dim=2)  # NxCxW
        x = x.permute(2, 0, 1)  # WxNxC
        x, _ = self.rnn(x)
        preds = self.clf(x)

        if self.training:
            rec_loss = self.recognition_loss_fn(preds, targets, self.voc_size)
            return preds, rec_loss
        else:
            # (W, N, C) -> (N, W, C)
            _, preds = preds.permute(1, 0, 2).max(dim=-1)
            return preds, None

### CoordConv
class MaskHead(nn.Module):
    def __init__(self, cfg):
        super(MaskHead, self).__init__()

        conv_dim = cfg.MODEL.BATEXT.CONV_DIM

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        convs = []
        convs.append(conv_block(258, conv_dim, 3, 1))
        for i in range(3):
            convs.append(conv_block(
                conv_dim, conv_dim, 3, 1))
        self.mask_convs = nn.Sequential(*convs)

    def forward(self, features):
        x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
        y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([features.shape[0], 1, -1, -1])
        x = x.expand([features.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_features = torch.cat([features, coord_feat], dim=1)
        mask_features = self.mask_convs(ins_features)
        return mask_features

def build_recognizer(cfg, type):
    if type == 'rnn':
        return RNNPredictor(cfg)
    if type == 'attn':
        return ATTPredictor(cfg)
    else:
        raise NotImplementedError("{} is not a valid recognizer".format(type))


def ctc_loss(preds, targets, voc_size):
    # prepare targets
    target_lengths = (targets != voc_size).long().sum(dim=-1)
    trimmed_targets = [t[:l] for t, l in zip(targets, target_lengths)]
    targets = torch.cat(trimmed_targets)

    x = F.log_softmax(preds, dim=-1)
    input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
    return F.ctc_loss(
        x, targets, input_lengths, target_lengths,
        blank=voc_size, zero_infinity=True
    )


def build_recognition_loss_fn(rec_type="ctc"):
    if rec_type == "ctc":
        return ctc_loss
    else:
        raise NotImplementedError("{} is not a valid recognition loss".format(rec_type))


@ROI_HEADS_REGISTRY.register()
class TextHead(nn.Module):
    """
    TextHead performs text region alignment and recognition.
    
    It is a simplified ROIHeads, only ground truth RoIs are
    used during training.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super(TextHead, self).__init__()
        # fmt: off
        pooler_resolution = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        pooler_scales     = cfg.MODEL.BATEXT.POOLER_SCALES
        sampling_ratio    = cfg.MODEL.BATEXT.SAMPLING_RATIO
        conv_dim          = cfg.MODEL.BATEXT.CONV_DIM
        num_conv          = cfg.MODEL.BATEXT.NUM_CONV
        canonical_size    = cfg.MODEL.BATEXT.CANONICAL_SIZE
        self.in_features  = cfg.MODEL.BATEXT.IN_FEATURES
        self.voc_size     = cfg.MODEL.BATEXT.VOC_SIZE
        recognizer        = cfg.MODEL.BATEXT.RECOGNIZER
        self.top_size     = cfg.MODEL.TOP_MODULE.DIM
        self.coordconv    = cfg.MODEL.BATEXT.USE_COORDCONV
        self.aet          = cfg.MODEL.BATEXT.USE_AET 
        # fmt: on

        self.pooler = TopPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="BezierAlign",
            canonical_box_size=canonical_size,
            canonical_level=3,
            assign_crit="bezier")

        conv_block = conv_with_kaiming_uniform(
            norm="BN", activation=True)
        tower = []
        for i in range(num_conv):
            tower.append(
                conv_block(conv_dim, conv_dim, 3, 1))
        self.tower = nn.Sequential(*tower)
       
        if self.coordconv: 
            self.mask_head = MaskHead(cfg)
        
        self.recognizer = build_recognizer(cfg, recognizer)
        self.dictionary = open("vn_dictionary.txt").read().replace("\n\n", "\n").split("\n")	
        self.trie = Trie(self.dictionary)

    def forward(self, images, features, proposals, targets=None):
        """
        see detectron2.modeling.ROIHeads
        """
        del images
        features = [features[f] for f in self.in_features]
        
        if self.coordconv:
            mask_features = []
            for i in range(len(features)):
                mask_feat = self.mask_head(features[i])
                all_feat = mask_feat + features[i]
                mask_features.append(all_feat)
            features = mask_features

        if self.training:
            beziers = [p.beziers for p in targets]
            if not self.aet:
                targets = torch.cat([x.text for x in targets], dim=0)
            else:
                beziers2 = [p.top_feat for p in proposals]
                for k in range(len(targets)):
                    rec_assign = [int(torch.argmin(torch.abs(beziers[k] - beziers2[k][i]).sum(dim=1))) for i in range(len(beziers2[k]))]
                    targets[k] = torch.cat([targets[k].text, targets[k].text[rec_assign]], dim = 0)
                targets = torch.cat([x for x in targets], dim = 0)
                cat_beziers = []
                for ix in range(len(beziers)):
                    cat_beziers.append(cat((beziers[ix], beziers2[ix]), dim=0))
                beziers = cat_beziers
                
            for target in targets:	
                rec = target.cpu().detach().numpy()	
                rec = decode(rec)	
                # candidates = {}	
                # candidates[rec] = 0	
                # for word in self.dictionary:	
                #     candidates[word] = eval(rec, word)	
                # candidates = sorted(candidates.items(), key=operator.itemgetter(1))[:10]	
                candidates_list = list(self.trie.all_levenshtein(rec, 1))	
                candidates_list.append(rec)	
                candidates_list = list(set(candidates_list))	
                candidates = {}	
                for candidate in candidates_list:	
                    candidates[candidate] = eval(rec, candidate)	
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))	
                dist_sharp = eval("###", rec)	
                while len(candidates) < 10:	
                    candidates.append(("###", dist_sharp))	
                candidates = candidates[:10]	
                candidates_encoded = []	
                distance_can = []	
                for can in candidates:	
                    word = []	
                    # print(can[0])	
                    for char in can[0]:	
                        word.append(CTLABELS.index(char))	
                    while len(word) < 25:	
                        word.append(104)	
                    word = word[:25]	
                    candidates_encoded.append(word)	
                    distance_can.append(1 / (can[1] + 0.1))	
                # distance_can = softmax(distance_can)	
                distance_candidates.append(distance_can)	
                target_candidates.append(candidates_encoded)	
            distance_candidates = torch.Tensor(distance_candidates).to(device="cuda")	
            # distance_candidates = torch.sum(distance_candidates, dim=0)	
            # distance_candidates = nn.functional.log_softmax(distance_candidates, dim=0)	
            target_candidates = torch.Tensor(target_candidates).to(device="cuda")	
            # distance_candidates = torch.Tensor(distance_candidates).to(device='cuda')	
            targets = target_candidates	
            targets = targets.permute((1, 0, 2))	
            targets = {"targets": targets, "scores": distance_candidates}	
            # for e in targets:	
            #     for e1 in e:	
            #         print(decode(e1))	
            #     print()
        else:
            beziers = [p.top_feat for p in proposals]
        bezier_features = self.pooler(features, beziers)
        bezier_features = self.tower(bezier_features)

        # TODO: move this part to recognizer
        if self.training:
            preds, rec_loss = self.recognizer(bezier_features, targets)
            rec_loss *= 0.05
            losses = {'rec_loss': rec_loss}
            return None, losses
        else:
            if bezier_features.size(0) == 0:
                for box in proposals:
                    box.beziers = box.top_feat
                    box.recs = box.top_feat
                return proposals, {}
            preds, _ = self.recognizer(bezier_features, targets)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.recs = preds[start_ind:end_ind]
                proposals_per_im.beziers = proposals_per_im.top_feat
                start_ind = end_ind
            return proposals, {}
