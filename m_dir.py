import numpy as np

class modified_dir:
    def __init__(self,dim,alpha,epsilon):
        self.dim = dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.count = np.zeros(self.dim)
        self.mode = None

        if alpha is not None:
            for i in range(self.dim):
                self.count[i] = self.alpha[i] - 1

        max_eps = 1./self.dim

        if epsilon > max_eps:
            # print "[mDir Warning] MDir(): espilon is too large, reset to the max value."
            self.epsilon = max_eps
        elif epsilon < 0:

            print "[mDir Warning] MDir(): espilon is negative, reset to 0."
            self.epsilon = 0
        else:
            self.epsilon = epsilon

    def get_mode(self):
        self.mode = np.zeros(self.dim)
        sum = 0.0
        n_eps = 0
        for i in range(self.dim):
            if self.count[i] > 0:
                self.mode[i] = self.count[i]
                sum += self.mode[i]
            else:
                self.mode[i] = -1
                n_eps +=1
        #if c <= 0 for all i
        if n_eps == self.dim:
            i_max = 0
            max = self.count[0]
            self.mode[0] = self.epsilon
            multi_modes = False
            for i in range(self.dim):
                if self.count[i] > max:
                    max = self.count[i]
                    i_max = i
                    multi_modes = False
                elif (self.count[i] == max):
                    multi_modes = True
                self.mode[i] = self.epsilon

            self.mode[i_max] = 1 - self.epsilon * (self.dim - 1)

            #warn if there are multiple equivalent modes
            # if (multi_modes):
            #     print "[mDir Warning] getMode(): multiple modes, returning one of them."
            return self.mode

        while(True):
            x = (1 - self.epsilon * n_eps) / sum
            sum = 0
            done = True
            n_eps = 0
            for i in range(self.dim):
                if self.mode[i] != -1:
                    self.mode[i] *= x
                    if self.mode[i] < self.epsilon:
                        self.mode[i] = -1
                        n_eps += 1
                        done = False
                    else:
                        sum += self.mode[i]
            if done:
                break

        for i in range(self.dim):
            if self.mode[i] == -1:
                self.mode[i] = self.epsilon

        return self.mode



