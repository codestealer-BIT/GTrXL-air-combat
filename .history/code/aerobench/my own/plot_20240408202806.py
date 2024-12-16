import matplotlib.pyplot as plt
import rl_utils
def plot(return_list,env_name,x_name,y_name,window_size):
    episodes_list = list(range(len(return_list)))
    plt.subplot(1,2,1)
    plt.plot(episodes_list, return_list)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 10)
    plt.subplot(1,2,2)
    plt.plot(episodes_list, mv_return)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('DQN on {}_moving average'.format(env_name))
    plt.show()