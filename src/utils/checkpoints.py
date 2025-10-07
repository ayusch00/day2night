import torch, os

def save_ckpt(model, opt, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),"opt": opt.state_dict(),"epoch": epoch}, path)

def load_ckpt(model, opt, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None: opt.load_state_dict(ckpt["opt"])
    return ckpt.get("epoch", 0)
