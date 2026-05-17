"""Path helpers for SSRS/RS3Mamba wire experiments."""
import os
import os.path as osp


def project_root():
    return osp.abspath(osp.join(osp.dirname(__file__), '..'))


def checkpoints1_dir():
    return osp.join(project_root(), 'data', 'checkpoints1')


def checkpoints2_dir():
    return osp.join(project_root(), 'data', 'checkpoints2')


def _has_wire_layout(path):
    return osp.isdir(osp.join(path, 'image', 'train')) and osp.isdir(
        osp.join(path, 'mask', 'train'))


def _ancestor_candidates():
    cur = project_root()
    for _ in range(12):
        yield cur
        parent = osp.dirname(cur)
        if parent == cur:
            break
        cur = parent


def resolve_data_ab_root():
    env = os.environ.get('WIRE_SEG_DATA_ROOT')
    if env and osp.isdir(env):
        return osp.abspath(env)
    for root in _ancestor_candidates():
        cand = osp.join(root, 'DataA-B')
        if osp.isdir(cand):
            return osp.abspath(cand)
    return osp.join(osp.dirname(project_root()), 'DataA-B')


def dataa_root():
    env = os.environ.get('WIRE_SEG_DATAA_ROOT')
    if env and osp.isdir(env):
        return osp.abspath(env)
    return osp.join(resolve_data_ab_root(), 'DataA')


def datab_root():
    env = os.environ.get('WIRE_SEG_DATAB_ROOT')
    if env and osp.isdir(env):
        return osp.abspath(env)
    return osp.join(resolve_data_ab_root(), 'DataB')


def datac_root():
    env = os.environ.get('WIRE_SEG_DATAC_ROOT')
    if env and osp.isdir(env):
        return osp.abspath(env)
    base = resolve_data_ab_root()
    candidates = [
        osp.join(base, 'DataC'),
        osp.join(osp.dirname(base), 'DataC'),
        osp.join(osp.dirname(base), 'DataC', 'DataC'),
    ]
    for root in _ancestor_candidates():
        candidates.extend([
            osp.join(root, 'DataC'),
            osp.join(root, 'DataC', 'DataC'),
        ])
    for cand in candidates:
        if _has_wire_layout(cand):
            return osp.abspath(cand)
    return osp.join(osp.dirname(base), 'DataC', 'DataC')


WIRE_SCHEMES = ('dataa', 'datab', 'datac')
_SUFFIX = {'dataa': 'A', 'datab': 'B', 'datac': 'C'}


def data_root_for_scheme(scheme):
    if scheme == 'dataa':
        return dataa_root()
    if scheme == 'datab':
        return datab_root()
    if scheme == 'datac':
        return datac_root()
    raise ValueError(f'unknown wire scheme: {scheme}')


def suffix_for_scheme(scheme):
    if scheme not in _SUFFIX:
        raise ValueError(f'unknown wire scheme: {scheme}')
    return _SUFFIX[scheme]


def image_size_for_scheme(scheme):
    return 512 if scheme == 'datac' else 256


def ensure_wire_dataset(scheme):
    root = data_root_for_scheme(scheme)
    if not _has_wire_layout(root):
        raise FileNotFoundError(
            f'Cannot find {scheme} dataset at {root}. Set WIRE_SEG_DATA_ROOT '
            'or WIRE_SEG_DATAA_ROOT / WIRE_SEG_DATAB_ROOT / WIRE_SEG_DATAC_ROOT.')
    return root
