import numpy as np
import os
import pickle
import time
import torch

from collections import deque
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from buffer import Buffer
from model import ActorCriticModel
from utils import batched_index_select, create_env, polynomial_decay, process_episode_info,average
from worker import Worker
from aerobench.visualize import anim3d, plot
from plot import plot
class PPOTrainer:
    def __init__(self, config:dict,filename, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set members
        self.config = config
        self.device = device
        self.run_id = run_id
        self.num_workers = config["n_workers"]
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]
        self.memory_length = config["transformer"]["memory_length"]
        self.num_blocks = config["transformer"]["num_blocks"]
        self.embed_dim = config["transformer"]["embed_dim"]
        self.filename=filename
        self.accumulated_res = {
            'status': [], 'times': [], 'states': [], 'modes': [],'missile':[],'final_state':[],
            'xd_list': [], 'ps_list': [], 'Nz_list': [], 'Ny_r_list': [], 'u_list': [], 'runtime': []
            }
        self.flag=True
        self.flag_1=True
        #self.flag=True#用于成功回合的单次可视化
        # Setup Tensorboard Summary Writer
        # if not os.path.exists("./summaries"):
        #     os.makedirs("./summaries")
        # timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        # self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment to retrieve action space, observation space and max episode length
        print("Step 1: Init dummy environment")
        dummy_env = create_env()#这里dummy_env只是为了把参数传给这么多space
        observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.n,)
        self.max_episode_length = dummy_env.max_episode_steps
        #dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, observation_space, self.action_space_shape, self.max_episode_length, self.device)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, self.action_space_shape, self.max_episode_length).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])#model里的参数一起更新

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.num_workers)]#这里才是环境的初始化
        self.worker_ids = range(self.num_workers)
        self.worker_current_episode_step = torch.zeros((self.num_workers, ), dtype=torch.long)
        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset",None,None))
        # Grab initial observations and store them in their respective placeholder location
        self.obs = np.zeros((self.num_workers,) + observation_space.shape, dtype=np.float32)
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

        # Setup placeholders for each worker's current episodic memory
        #self.memory的结构和self.buffer.memory不一样，还待后续观察
        self.memory = torch.zeros((self.num_workers, self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)
        # Generate episodic memory mask used in attention
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """         
        # Setup memory window indices to support a sliding window over the episodic memory
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(self.max_episode_length - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:主要是前四步只能用一样的
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """
        #比方说memorylength=4 目前在第3步，在共同作用下会选择，memory_indices为0120的,也就是前3步的记忆。目前在第50步的话，会选择47 48 49 0。
        #掩码在回合步数大于memorylength之后就失效了，因为这里前memorylength步的记忆都可被利用，没有掩码了。
    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model. Only the final model is saved."""
        print("Step 6: Starting training using " + str(self.device))
        # Store episode results for monitoring statistics
        episode_infos = []#会把老元素挤出去

        for update in range(self.config["updates"]):#200轮
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            # beta=0.0001
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # Sample training data
            sampled_episode_info = self._sample_training_data(update)

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats, grad_info = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = process_episode_info(episode_infos)

            # Print training statistics
            success_count = sum(1 for info in sampled_episode_info if info.get("success") == True)
            returns=np.mean([info['reward'] for info in sampled_episode_info])
            result = "{:4} return={:.2f} std={:.2f} length_update={:.1f} length={:.1f} std={:.2f} win_num_update={:.1f} win_rate={:.2f}% pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update,returns, episode_result["reward_std"], len(sampled_episode_info),episode_result["length_mean"], episode_result["length_std"],success_count,episode_result["success_percent"]*100,
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            # if "success" in episode_result:
            #     result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
            #         update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success"],
            #         training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            # else:
            #     result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
            #         update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
            #         training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            print(result)
        #这一段得放到writer_summary的前面，不然输出顺序是乱的
        success_numeric = [1 if entry['success'] else 0 for entry in episode_infos]
        rewards=[entry["reward"] for entry in episode_infos]
        with open("output.txt", "w") as file:
            print(average(success_numeric), file=file)
            print(average(rewards), file=file)
        
        # for w, worker in enumerate(self.workers):
        #     #child是子进程
        #     worker.child.send(("tra",w,None,None))

        #     # Write training statistics to tensorboard
        #     self._write_gradient_summary(update, grad_info)
        #     self._write_training_summary(update, training_stats, episode_result)
        
        # Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self,update) -> list:
        """Runs all n workers for n steps to sample training data.
        作用：智能体根据model返回policy、value，然后存在self.buffer里，环境根据动作更新奖励、obs、dones、info
        如果一个回合结束（根据info判断），则更新memory、状态、memory_index，回合步数eps_step
        最后计算优势函数存在self.buffer.advantages，用于PPO策略更新
        Returns:
            {list} -- list of results of completed episodes.
            info = {"success": success,
                    "reward": sum(self._rewards),
                    "length": len(self._rewards)}
        """
        episode_infos = []
        # Init episodic memory buffer using each workers' current episodic memory
        self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]#在这初始化,后续如果self.memory产生变化，那么self.buffer.memories也会变化
        for w in range(self.num_workers):#memory_index赋值
            self.buffer.memory_index[w] = w

        # Sample actions from the model and collect experiences for optimization
        for t in range(self.config["worker_steps"]):#一个回合的时间步
            # Gradients can be omitted for sampling training data

            with torch.no_grad():
                # Store the initial observations inside the buffer
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                # Store mask and memory indices inside the buffer
                self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]#shape:n_workers,memory_length
                self.buffer.memory_indices[:, t] = self.memory_indices[self.worker_current_episode_step]#shape:n_workers,memory_length
                # Retrieve the memory window from the entire episodic memory
                sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])#num_workers,memory_length,num_blocks,embed_dim
                # Forward the model to retrieve the policy, the s' value and the new memory item
                policy, value, memory = self.model(torch.tensor(self.obs), sliced_memory, self.buffer.memory_mask[:, t],
                                                   self.buffer.memory_indices[:,t])
                
                # Add new memory item to the episodic memory
                self.memory[self.worker_ids, self.worker_current_episode_step] = memory#num_workers,num_blocks,embed_dim,代表着每个worker在当前时间步的记忆将由memory更新
                #self.memory在回合中不断向前rollout迭代，每迭代一次就填充上对应t时间步的num_workers,num_blocks,embed_dim,100步结束后self.memory被填满，将传给self.buffer.memory

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)#分支对应并行运算，按列堆叠
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                #child负责send，终端负责recv
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy(),int(self.worker_current_episode_step[w])))#env的step函数返回的应该是一个yield
            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t],info,res = worker.child.recv()
                if info: # i.e. done
                    # Reset the worker's current timestep
                    self.worker_current_episode_step[w] = 0
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)#这个东西的索引不能和w挂钩，它是谁先输谁放第一个
                    # Reset the agent (potential interface for providing reset parameters)
                    worker.child.send(("reset",None,None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Break the reference to the worker's memory
                    mem_index = self.buffer.memory_index[w, t]#标量，mem_index==w?
                    self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()#self.buffer.memories = [self.memory[w] for w in range(self.num_workers)]，深克隆，新的变量里的东西和self.memory无关了
                    # Reset episodic memory
                    self.memory[w] = torch.zeros((self.max_episode_length, self.num_blocks, self.embed_dim), dtype=torch.float32)#记忆不用了，清空记忆
                    if t < self.config["worker_steps"] - 1:
                        # Store memory inside the buffer
                        self.buffer.memories.append(self.memory[w])
                        # Store the reference of to the current episodic memory inside the buffer
                        self.buffer.memory_index[w, t + 1:] = len(self.buffer.memories) - 1#这个worker在t时刻之后都用t时刻的memory，这个memory_index就是在self.buffer.memories里的位置
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1
                # Store latest observations
                self.obs[w] = obs
        for w, _ in enumerate(self.workers):#重置step，为了m_state正确传入
            self.worker_current_episode_step[w]=0
            worker.child.send(("reset",None,None))
            self.obs[w] = worker.child.recv()
                            
        # Compute the last value of the current observation and memory window to compute GAE
        last_value = self.get_last_value()#这里self.obs已经是最后一个状态了
        # Compute advantages
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def get_last_value(self):
        """Returns:
                {torch.tensor} -- Last value of the current observation and memory window to compute GAE"""
        start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_step, self.memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.num_workers)]).long()
        sliced_memory = batched_index_select(self.memory, 1, indices) # Retrieve the memory window from the entire episode
        _, last_value, _ = self.model(torch.tensor(self.obs),
                                        sliced_memory, self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        self.buffer.memory_indices[:,-1])
        return last_value

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {tuple} -- Training and gradient statistics of one training epoch"""
        train_info, grad_info = [], {}
        for _ in range(self.config["epochs"]):#4，训练4大轮
            mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:#mini_batch大小是num_workers*worker_steps//n_mini_batch(16*128//8)
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
                for key, value in self.model.get_grad_norm().items():
                    grad_info.setdefault(key, []).append(value)
        return train_info, grad_info#返回的是训练信息和模型梯度信息

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of training statistics (e.g. loss)
        """
        # Select episodic memory windows
        memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
        
        # Forward model
        policy, value, _ = self.model(samples["obs"], memory, samples["memory_mask"], samples["memory_indices"])
        #这个policy是什么,每一个状态对应动作的概率分布
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)
        # loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss )
        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        # Monitor additional training stats
        approx_kl = (ratio - 1.0) - log_ratio # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > clip_range).float().mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.mean().cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)
        self.writer.add_scalar("other/clip_fraction", training_stats[4], update)
        self.writer.add_scalar("other/kl", training_stats[5], update)
        
    def _write_gradient_summary(self, update, grad_info):
        """Adds gradient statistics to the tensorboard event file.

        Arguments:
            update {int} -- Current PPO Update
            grad_info {dict} -- Gradient statistics
        """
        for key, value in grad_info.items():
            self.writer.add_scalar("gradients/" + key, np.mean(value), update)

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close",None,None))
        except:
            pass

        time.sleep(1.0)
        exit(0)