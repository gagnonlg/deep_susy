import collections

import numpy as np


def __sr(dset, cond, mg=None, ml=None):

    #tot = np.sum(dset['M_weight'])

    if mg is not None and ml is not None:
        cond = [
            'I_m_gluino == {}'.format(mg),
            'I_m_lsp == {}'.format(ml),
        ] + cond

    array = dset
    for c in cond:
        fields = c.split(' ')
        fields[0] = "array['" + fields[0] + "']"
        c = ' '.join(fields)
        array = array[:][np.where(eval(c))]

    return (
        # tot,
        np.sum(array['M_weight']),
        np.sqrt(np.sum(array['M_weight'] ** 2))
    )


__gtt_1l_B = [
    'M_nlepton >= 1',
    'M_nb77 >= 3',
    'M_njet30 >= 5',
    'M_mt > 150',
    'M_mtb > 120',
    'M_met > 500',
    'M_meff > 2200',
    'M_mjsum > 200',
]

def gtt_1l_B(dset, mg=None, ml = None):
    return __sr(dset, __gtt_1l_B, mg, ml)

__gtt_1l_M = [
    'M_nlepton >= 1',
    'M_nb77 >= 3',
    'M_njet30 >= 6',
    'M_mt > 150',
    'M_mtb > 160',
    'M_met > 450',
    'M_meff > 1800',
    'M_mjsum > 200',
]

def gtt_1l_M(dset, mg=None, ml=None):
    return __sr(dset, __gtt_1l_M, mg, ml)

__gtt_1l_C = [
    'M_nlepton >= 1',
    'M_nb77 >= 3',
    'M_njet30 >= 7',
    'M_mt > 150',
    'M_mtb > 160',
    'M_met > 350',
    'M_meff > 1000',
]

def gtt_1l_C(dset, mg=None, ml=None):
    return __sr(dset, __gtt_1l_C, mg, ml)

__gtt_0l_B = [
    'M_nlepton == 0',
    'M_nb77 >= 3',
    'M_njet30 >= 7',
    'M_mtb > 60',
    'M_met > 350',
    'M_meff > 2600',
    'M_mjsum > 300',
]

def gtt_0l_B(dset, mg=None, ml=None):
    return __sr(dset, __gtt_0l_B, mg, ml)

__gtt_0l_M = [
    'M_nlepton == 0',
    'M_nb77 >= 3',
    'M_njet30 >= 7',
    'M_mtb > 120',
    'M_met > 500',
    'M_meff > 1800',
    'M_mjsum > 200',
]

def gtt_0l_M(dset, mg=None, ml=None):
    return __sr(dset, __gtt_0l_M, mg, ml)

__gtt_0l_C = [
    'M_nlepton == 0',
    'M_nb77 >= 4',
    'M_njet30 >= 8',
    'M_mtb > 120',
    'M_met > 250',
    'M_meff > 1000',
    'M_mjsum > 100',
]

def gtt_0l_C(dset, mg=None, ml=None):
    return __sr(dset, __gtt_0l_C, mg, ml)

SR_dict = collections.OrderedDict([
    ('gtt_0l_B', gtt_0l_B),
    ('gtt_0l_M', gtt_0l_M),
    ('gtt_0l_C', gtt_0l_C),
    ('gtt_1l_B', gtt_1l_B),
    ('gtt_1l_M', gtt_1l_M),
    ('gtt_1l_C', gtt_1l_C)
])
