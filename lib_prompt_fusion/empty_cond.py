from lib_prompt_fusion import interpolation_tensor


_empty_cond = None


def get():
    return _empty_cond


def init(model):
    global _empty_cond
    cond = model.get_learned_conditioning([''])
    if isinstance(cond, dict):
        cond = interpolation_tensor.DictCondWrapper({k: v[0] for k, v in cond.items()})
    else:
        cond = interpolation_tensor.TensorCondWrapper(cond[0])

    _empty_cond = cond
