"""Explicit two-class helmet adapter for separately evaluated candidate models.

Only the inspected Hardhat / NO-Hardhat mapping is accepted. No absence rule.
This module does not replace or reconfigure the working SafetyVision TPU.
"""
import ast

import numpy as np

from gateways.safetyvision.decoder import safetyvision_detections


def validate_helmet_session(session):
    inputs,outputs = session.get_inputs(),session.get_outputs()
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names','{}'))
    if (len(inputs)!=1 or inputs[0].name!='images' or inputs[0].shape!=[1,3,640,640] or
            len(outputs)!=1 or outputs[0].shape!=[1,6,8400] or names!={0:'Hardhat',1:'NO-Hardhat'}):
        raise ValueError('Helmet candidate must have verified Hardhat / NO-Hardhat [1,6,8400] contract')


def helmet_detections(raw,meta):
    raw = np.asarray(raw)
    if raw.shape!=(1,6,8400):
        raise ValueError('Unexpected helmet candidate output')
    # Reuse the audited coordinate transform, single-label policy and NMS. The
    # adapter only remaps the two verified class-score channels, not their scale.
    lifted = np.zeros((1,17,8400),dtype=raw.dtype)
    lifted[:,:4] = raw[:,:4]
    lifted[:,7] = raw[:,4]  # Hardhat -> SafetyVision index 3 (+4 box channels)
    lifted[:,11] = raw[:,5]  # NO-Hardhat -> canonical no_helmet
    return safetyvision_detections(lifted,meta,class_thresholds={'no_helmet':.45,'hardhat':.40})
