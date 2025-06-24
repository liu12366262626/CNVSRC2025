#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import json
import torch
import argparse
import numpy as np

from ..espnet.asr.asr_utils import torch_load
from ..espnet.asr.asr_utils import get_model_conf
from ..espnet.asr.asr_utils import add_results_to_json
from ..espnet.nets.batch_beam_search import BatchBeamSearch
from ..espnet.nets.lm_interface import dynamic_import_lm
from ..espnet.nets.scorers.length_bonus import LengthBonus
from ..espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

from .data.transforms import TextTransform

class AVSR(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)


    def load_checkpoint(self):
        if self.cfg.ckpt_path:
            ckpt = torch.load(
                self.cfg.ckpt_path, map_location=lambda storage, loc: storage
            )
            if self.cfg.transfer_frontend:
                tmp_ckpt = {
                    k: v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("trunk.") or k.startswith("frontend3D.")
                }
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            else:
                if self.cfg.remove_ctc:
                    dict_new = {}
                    for key, value in ckpt.items():
                        if key not in ['decoder.embed.0.weight', 'decoder.output_layer.weight', 'decoder.output_layer.bias',
                                       'r_decoder.embed.0.weight', 'r_decoder.output_layer.weight', 'r_decoder.output_layer.bias',
                                       'ctc.ctc_lo.weight', 'ctc.ctc_lo.bias']:
                            dict_new[key] = value
                    self.model.load_state_dict(dict_new, strict=False)
                else:
                    if self.cfg.ckpt_path.endswith('pth'):
                        new_ckpt = {}
                        for key, value in ckpt.items():
                            new_ckpt["model." + key] = value
                        self.load_state_dict(new_ckpt)
                    else:
                        self.load_state_dict(ckpt["state_dict"])

    def infer(self, data):

        self.model.eval()
        

        with torch.no_grad():
            beam_search = get_beam_search_decoder(self.model, self.token_list, ctc_weight=0.3,)
            enc_feat, _ = self.model.encoder(data.unsqueeze(0), None)
            enc_feat = enc_feat.squeeze(0)
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted = add_results_to_json(nbest_hyps, self.token_list)
            transcript = predicted.replace("▁", " ").strip().replace("<eos>", "")
        return transcript


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
