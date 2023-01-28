_empty_cond = None


def get():
    return _empty_cond


def init_empty_embedding(model):
    global _empty_cond
    if _empty_cond is None:
        _empty_cond = model.get_learned_conditioning([''])[0]
