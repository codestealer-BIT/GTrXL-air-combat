import numpy as np
import pickle
import torch

from docopt import docopt
from model import ActorCriticModel
from F_16env_trxl import F_16env_trxl
from aerobench.visualize import anim3d
def init_transformer_memory(trxl_conf, max_episode_steps, device):
    """Returns initial tensors for the episodic memory of the transformer.

    Arguments:
        trxl_conf {dict} -- Transformer configuration dictionary
        max_episode_steps {int} -- Maximum number of steps per episode
        device {torch.device} -- Target device for the tensors

    Returns:
        memory {torch.Tensor}, memory_mask {torch.Tensor}, memory_indices {torch.Tensor} -- Initial episodic memory, episodic memory mask, and sliding memory window indices
    """
    # Episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    # Episdic memory tensor
    memory = torch.zeros((1, max_episode_steps, trxl_conf["num_blocks"], trxl_conf["embed_dim"])).to(device)
    # Setup sliding memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(max_episode_steps - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Set inference device and default tensor type
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = F_16env_trxl(episodes=50000,filename='')

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,), env.max_episode_steps)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []
    memory, memory_mask, memory_indices = init_transformer_memory(config["transformer"], env.max_episode_steps, device)
    memory_length = config["transformer"]["memory_length"]
    t = 0
    obs = env.reset()#这里只有一个worker
    accumulated_res = {
    'status': [], 'times': [], 'states': [], 'modes': [],'missile':[],'final_state':[],
    'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
    }
    while not done:
        # Prepare observation and memory
        obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
        in_memory = memory[0, memory_indices[t].unsqueeze(0)]
        t_ = max(0, min(t, memory_length - 1))
        mask = memory_mask[t_].unsqueeze(0)
        indices = memory_indices[t].unsqueeze(0)
        # Render environment
        # Forward model
        policy, value, new_memory = model(obs, in_memory, mask, indices)
        memory[:, t] = new_memory
        # Sample action
        action = []
        for action_branch in policy:
            action.append(action_branch.sample().item())
        # Step environemnt
        obs, reward, done, info,res = env.step(action[-1],None if t==0 else m_state,t)#action是一个列表，输入应该是一个数
        m_state=res["final_state"]
        episode_rewards.append(reward)
        t += 1
        accumulated_res['status'] = res['status']
        accumulated_res['times'].extend(res['times'])
        accumulated_res['states'].append(res['states'])
        accumulated_res['missile'].append(res['missile'])
        accumulated_res['modes'].extend(res['modes'])
        accumulated_res['final_state'].append(res['final_state'])
        if 'xd_list' in res:
            accumulated_res['xd_list'].extend(res['xd_list'])
            accumulated_res['ps_list'].extend(res['ps_list'])
            accumulated_res['Nz_list'].extend(res['Nz_list'])
            accumulated_res['Ny_r_list'].extend(res['Ny_r_list'])
            accumulated_res['u_list'].extend(res['u_list'])
        accumulated_res['runtime'].append(res['runtime'])
    accumulated_res['states'] = np.vstack(accumulated_res['states'])
    accumulated_res['missile'] = np.vstack(accumulated_res['missile'])
    anim3d.make_anim(accumulated_res, filename='', f16_scale=70, viewsize=60000, viewsize_z=4000, trail_pts=np.inf,
                elev=27, azim=-107, skip_frames=15,
                chase=True, fixed_floor=False, init_extra=None)

    env.close()

if __name__ == "__main__":
    main()