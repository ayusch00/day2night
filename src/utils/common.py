import yaml, torch, random, numpy as np, os, time

def load_cfg(path): return yaml.safe_load(open(path))
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def make_run_dirs(cfg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run = f"{cfg['exp_name']}_{ts}"
    exp_dir = os.path.join(cfg["logging"]["out_dir"], run)
    res_dir = cfg["logging"]["results_dir"]
    os.makedirs(exp_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    return exp_dir, res_dir
