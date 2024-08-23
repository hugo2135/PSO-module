import numpy as np
from time import time

class PSO():
    def __init__(self, loss, c1, c2, x_max, x_min, num_particle, num_parameter, max_iter,
                 k = 0.9, threshold=0.001, w_max = 1.0, w_min = 0.1):
        
        np.random.seed(int(time()))
        #外部Input
        self.c1 = c1
        self.c2 = c2
        self.x_max =x_max*np.ones(num_parameter)
        self.x_min =x_min*np.ones(num_parameter)
        self.k = k
        self.num_particle = num_particle
        self.num_parameter = num_parameter
        self.max_iter = max_iter
        self.threshold = threshold
        self.w_max = w_max
        self.w_min = w_min
        self.loss = loss
    
        #粒子群相關
        self.X = np.random.uniform(low=self.x_min, high=self.x_max, size=[self.num_particle, self.num_parameter])
        self.V = np.zeros(shape=[self.num_particle, self.num_parameter])
        self.temp_V = np.zeros(shape=[self.num_particle, self.num_parameter])
        self.v_max = self.k*(self.x_max-self.x_min)/2
        self.individual_best_solution = self.X.copy()
        self.individual_best_value = np.ndarray(self.num_particle)
        for i in range(self.num_particle):
            self.individual_best_value[i] = self.loss(self.X[i])
        self.global_best_solution = self.individual_best_solution[self.individual_best_value.argmin()]
        self.global_best_value = self.individual_best_value.min()

        #記錄用
        self.X_history = []
        self.V_history = []
        self.global_best_solution_history = []
        self.log = ""

    def PSO(self):
        _iter = 0
        while (_iter < self.max_iter) and abs(self.global_best_value) > self.threshold:
            self.X_history.append(self.X.copy())
            self.V_history.append(self.V.copy())
            R1 = np.random.uniform(size=(self.num_particle, self.num_parameter))
            R2 = np.random.uniform(size=(self.num_particle, self.num_parameter))
            w = self.w_max - _iter*(self.w_max-self.w_min)/self.max_iter

            for i in range(self.num_particle):
                #速度變化方程式
                self.V[i, :] = w*self.V[i, :] + self.c1*(self.individual_best_solution[i,:] - self.X[i,:])*R1[i,:] + self.c2*(self.global_best_solution - self.X[i,:])*R2[i,:]
                
                #速度限制
                self.V[i, self.v_max < self.V[i, :]] = self.v_max[self.v_max < self.V[i, :]]
                self.V[i, -self.v_max > self.V[i, :]] = -self.v_max[-self.v_max > self.V[i, :]]
                #位置更新
                self.X[i, :] = self.X[i, :] + self.V[i, :]
                self.X[i, self.x_max < self.X[i, :]] = self.x_max[self.x_max < self.X[i, :]]
                self.X[i, self.x_min > self.X[i, :]] = self.x_min[self.x_min > self.X[i, :]]
            
            self.V_history.append(self.V)
            for i in range(self.num_particle):
                score = self.loss(self.X[i])
                if abs(score)<abs(self.individual_best_value[i]):
                    self.individual_best_value[i] = score
                    self.individual_best_solution[i, :] = self.X[i, :].copy()
                    if abs(score)<abs(self.global_best_value):
                        self.global_best_value = score
                        self.global_best_solution = self.X[i, :].copy()
            self.global_best_solution_history.append(self.global_best_solution)
            print("<---------------------->\n"+\
                    f"Iteration{_iter:02d}:\n"+\
                    f"   |--Error:{score}\n"+\
                    "----------------------")
            _iter += 1