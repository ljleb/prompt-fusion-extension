_empty_cond = None


def get():
    return _empty_cond


def init(model):
    global _empty_cond
    if _empty_cond is None:
        cond_res = model.get_learned_conditioning([''])
        if isinstance(cond_res, dict):
            cond_res = cond_res['crossattn']
        _empty_cond = cond_res[0]
