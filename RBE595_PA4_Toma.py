'''
Shane Toma | RBE595 Programming Assignment 4
'''
import numpy as np
import matplotlib.pyplot as plt
#setup cliff environment
class CliffEnv:

    def __init__(self):
        #init world, goals and states
        self.world = np.loadtxt("cliff_world.txt")
        self.start = (3,0)
        self.goal = (3,11)
        self.max_y, self.max_x = self.world.shape
        self.goal_y, self.goal_x = self.goal
        self.states = list(map(tuple, np.ndindex(self.world.shape)))
        self.n_states = len(self.states)
        self.n_actions = 4

        #create dict for action states
        self.actions = {
            "up":(-1,0),
            "down":(1,0),
            "left":(0,-1),
            "right":(0,1)
        }

    def calc_next_state(self,state,action):
        #calc new state based on current state and action
        next_state = tuple(map(sum, zip(state, self.actions[action])))
       
        if self.world[next_state] == 1:
            #print("Whoops, you fell off the cliff. Returning to start")
            next_state = self.start

        return next_state
    
    def calc_reward(self,state):

        if self.world[state] == 1:
            self.reward = -1000
            
        else:
            self.reward = -1
        
        return self.reward

class TD:
    def __init__(self, environment, epsilon, gamma, alpha) -> None:

        self.cliff_env = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
    
    def init_Q(self):
        Q={}
        for state in self.cliff_env.states:
            Q[state] = {action:0 for action in self.valid_actions(state)}
        return Q
    
    #remove actions which will move to out of bounds states
    def valid_actions(self,state):
        valid_actions = []
        for action, (dx,dy) in self.cliff_env.actions.items():
            next_state = (state[0] + dx, state[1] + dy)
            if self.is_valid_state(next_state):
                valid_actions.append(action)
        return valid_actions
    #check if action leads to state within gridworld
    def is_valid_state(self,state):
        return 0 <= state[0] < self.cliff_env.world.shape[0] and 0 <= state[1] < self.cliff_env.world.shape[1]

    def Q_learning(self, n_episodes):

        Q = self.init_Q()
        total_rewards = []

        for episode in range(n_episodes):
            cum_reward = 0
            state = self.cliff_env.start

            #loop for each step in an episode
            while True:
                e_greedy = np.random.rand()
                #choose action using epsilon greedy
                if e_greedy <= self.epsilon: #greedy actions
                    action = np.random.choice(list(Q[state]))
                else:
                    action = max(Q[state], key=Q[state].get, default=None)
                #get reward, next state from action, S'
                next_state = self.cliff_env.calc_next_state(state,action)

                if next_state == self.cliff_env.start:
                    reward = -100 #set reward to 100 if state is reset by cliff fall
                else:
                    reward = -1

                cum_reward += reward

                Q[state][action] += self.alpha*(reward+self.gamma*max(Q[next_state].values())-Q[state][action]) 
                state = next_state

                if self.cliff_env.world[state] == 1 or state == self.cliff_env.goal:
                    break
                
                
            total_rewards.append(cum_reward)
            

        return Q, total_rewards

    def SARSA(self, n_episodes):

        #initialize Q(s,a) for all states, actions
        Q = self.init_Q()
        total_reward = []
        
        #loop for every episode
        for episode in range(n_episodes):
            cum_reward = 0
            state = self.cliff_env.start #starting state
            valid = 0
            #loop for each step in an episode
            while True:
                e_greedy = np.random.rand()
                
                #choose action using epsilon greedy
                if e_greedy <= self.epsilon: #greedy actions
                    action = np.random.choice(list(Q[state]))
                else:
                    
                    action = max(Q[state], key=Q[state].get, default=None)
                #get reward, next state from action, S'
                next_state = self.cliff_env.calc_next_state(state,action)

                if next_state == self.cliff_env.start:
                    reward = -100 #set reward to 100 if state is reset by cliff fall
                else:
                    reward = -1

                cum_reward += reward
                #choose A' from S' with policy from Q
                if np.random.rand() <= self.epsilon: #greedy actions
                    next_action = np.random.choice(list(Q[next_state]))
                else:
                    next_action = max(Q[next_state], key=Q[next_state].get)
                #print(Q[next_state][next_action])
                Q[state][action] = Q[state][action] + self.alpha*(reward+self.gamma*Q[next_state][next_action]-Q[state][action] )

                state = next_state
                action = next_action
                #end episode if these conditions are met
                if self.cliff_env.world[state] == 1 or state == self.cliff_env.goal:
                    break
                
                
            total_reward.append(cum_reward)
            

        return Q, total_reward
    
    def plot_path(self,Q,start,goal,line,_label):

        start_x, start_y = start
        goal_x, goal_y = goal
        plt.plot(start_y,start_x,"go")
        plt.plot(goal_y,goal_x,"ro")
        plt.imshow(self.cliff_env.world, cmap="binary")

        path = []
        state = start
        #find the path of optimal states and actions
        while True:
            path.append(state)
            opt_action = max(Q[state], key=Q[state].get, default=None)
            path.append(opt_action)
            state = self.cliff_env.calc_next_state(state,opt_action)

            if state == goal:
                break
            y = []
            x = []
        #draw line for each coordinate pair in path
        for i in range(0,len(path),2):
            state, opt_action = path[i],path[i+1]
            y.append(state[0])
            x.append(state[1])
            dy,dx = self.cliff_env.actions[opt_action]
        y.append(goal_x)
        x.append(goal_y)
        plt.plot(x,y,line,label = _label)
        #plt.show()
                

# main code- produce plots
cliff = CliffEnv()
td = TD(cliff,0.0001, 0.9, 0.2)

Q_S, tot_r_S = td.SARSA(1000)
Q_Q, tot_r_Q = td.Q_learning(1000)
episodes = [i for i in range(1000)]

running_avg_S = np.cumsum(tot_r_S)/episodes
running_avg_Q = np.cumsum(tot_r_Q)/episodes

plt.plot(episodes,running_avg_Q)
plt.plot(episodes,running_avg_S)
plt.title("Sum of Episodic Rewards Over 1000 Episodes")
plt.legend(["Q Learning", "SARSA"])
plt.ylim((-400,0))
plt.show()



td.plot_path(Q_S,cliff.start,cliff.goal,"-b", "SARSA")
td.plot_path(Q_Q,cliff.start,cliff.goal,"--r","Q Learning")
plt.legend()
plt.title("Cliff Walk Performance for Learning Methods, $\epsilon$=0.0001")

plt.show()


