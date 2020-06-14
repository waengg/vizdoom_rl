class PrioritizedReplay:
    def __init__(self, maxlen=1000):
        self.max_memory = max_memory
        self.memory = []
        self.store_episodes = store_episodes
        self.current_index = 0

        # prioritized experience replay params
        self.prioritized = prioritized
        self.alpha = 0.6 # prioritization factor
        self.beta_start = 0.4
        self.beta_end = 1
        self.beta = self.beta_end
        self.sum_powered_priorities = 0 # sum p^alpha

    
    def enqueue(self, state, isterminal, priority):
        """Add a transition to the experience replay
        :param transition: the transition to insert
        :param game_over: is the next state a terminal state?
        """
        # set the priority to the maximum current priority
        transition_powered_priority = 1e-7 ** self.alpha
        if self.prioritized:
            transition_powered_priority = np.max(self.memory,1)[0,2]
        self.sum_powered_priorities += transition_powered_priority

        # store transition
        if self.is_last_record_closed():
            self.add_record(transition, game_over, transition_powered_priority)
        else:
            self.get_last_record().add_transition(transition, game_over, transition_powered_priority)
        # finalize the record if necessary
        if not self.store_episodes or (self.store_episodes and (game_over or transition.reward > 0)): #TODO: this is wrong
            self.close_last_record()

        # free some space (delete the oldest transition or episode)
        if len(self.memory) > self.max_memory:
                    self.sum_powered_priorities -= self.memory[0].transition_powered_priority
            del self.memory[0]


    def get_batch(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        if self.prioritized: # TODO: not currently working for episodic experience replay
            # prioritized experience replay
            probs = np.random.rand(batch_size)
            importances = [self.get_transition_importance(idx) for idx in range(len(self.memory))]
            thresholds = np.cumsum(importances)

            # multinomial sampling according to priorities
            indices = []
            for p in probs:
                for idx, threshold in enumerate(thresholds):
                    if p < threshold:
                        indices += [idx]
                        break
        else:
            indices = np.random.choice(len(self.memory), batch_size)

        minibatch = list()
        for idx in indices:
            while not_terminals and self.memory[idx].game_over:
                idx = np.random.choice(len(self.memory), 1)[0]
            weight = self.get_transition_weight(idx)
            minibatch.append([idx, self.memory[idx].transition_list, self.memory[idx].game_over, weight])  # idx, [transition, transition, ...] , game_over, weight

        if self.prioritized:
            max_weight = np.max(minibatch,0)[3]
            for idx in range(len(minibatch)):
                minibatch[idx][3] /= float(max_weight) # normalize weights relative to the minibatch

        return minibatch

    def get_transition_weight(self, transition_idx):
        """Get the weight of a transition by its index
        :param transition_idx: the index of the transition
        :return: the weight of the transition - 1/(importance*N)^beta
        """
        weight = 1/float(self.get_transition_importance(transition_idx)*self.max_memory)**self.beta
        return weight

    def get_transition_importance(self, transition_idx):
        """Get the importance of a transition by its index
        :param transition_idx: the index of the transition
        :return: the importance - priority^alpha/sum(priority^alpha)
        """
        powered_priority = self.memory[transition_idx]['transition_powered_priority']
        importance = powered_priority / float(self.sum_powered_priorities)
        return importance

    pass