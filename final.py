import numpy as np
import matplotlib.pyplot as plt
import random

class Q1:
    def __init__(self):
        self.eq = lambda x, y: np.exp(np.pow(x,6) - np.pow(y,4))
    
    def solution1(self, N = 10 ** 6):
        total_sum = 0
        for _ in range(N):
            x,y = random.uniform(-1,1), random.uniform(-1,1)
            total_sum += self.eq(x,y)
        val = total_sum / N
        print(f"Approximated integral value through random uniform sampling: {val}, error: {val - 4.028423}")
        
        
    def compute_gradient(self, x,y):
        return np.array([6 * np.pow(x,5) * np.exp(np.pow(x,6) - np.pow(y,4)), -4 * np.pow(y, 3) * np.exp(np.pow(x,6) - np.pow(y,4))]) 
      
    def get_probability_distribution(self, N=100):
      x_vals = np.linspace(-1, 1, N)
      y_vals = np.linspace(-1, 1, N)
      X , Y = np.meshgrid(x_vals, y_vals)

      gradient_magnitudes = []
      for a, b in zip(X.ravel(), Y.ravel()):
        gx, gy = self.compute_gradient(a, b)
        gradient_magnitudes.append(np.sqrt(gx ** 2 + gy ** 2))

      gradient_magnitudes = np.array(gradient_magnitudes)
      probability_distribution = gradient_magnitudes / np.sum(gradient_magnitudes)

      return probability_distribution.reshape(N, N)

    def solution2(self, N=int(1e6), grid_size=100):
      dist = self.get_probability_distribution(grid_size)  # shape (grid_size, grid_size)
      dx = 1 / grid_size
      dy = 1 / grid_size
      cell_area = dx * dy

      flat_probs = dist.ravel()
      total_cells = grid_size ** 2
      coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    # Choose cells according to gradient-based PMF
      chosen_indices = np.random.choice(np.arange(total_cells), size=N, p=flat_probs)

      total_sum = 0

      for index in chosen_indices:
        i, j = coords[index]
        # Sample uniformly within cell [i*dx, (i+1)*dx] × [j*dy, (j+1)*dy]
        x = (i + random.random()) * dx
        y = (j + random.random()) * dy

        p_ij = dist[i][j]
        if p_ij == 0:
            continue  # skip this sample (should rarely happen)

        # Importance sampling weight
        weight = cell_area / p_ij
        total_sum += self.eq(x, y) * weight

      val = total_sum / N
      print(f"Integral value by method 2: {val}, error: {val - 4.028423}")
class Q3:
    def __init__(self):
        self.diffeq = lambda x_deriv, x, t: np.exp(-x_deriv) - x_deriv - np.pow(x, 3) + (3 * np.exp(-np.pow(t,3))) # Implicit form of differential equation
        self.x_initial, self.t_initial = 1, 0
        
    def compute_derivative(self, x, equation):
        step_size = 0.01
        return (equation(x + step_size) - equation(x - step_size)) / (2 * step_size)
        
    def newton_raphson(self, x, t):
        x_deriv_val = 2
        equation = lambda x_deriv: self.diffeq(x_deriv, x, t)
        tolerance = 10 ** (-4)
        while abs(equation(x_deriv_val)) > tolerance:
            x_deriv_val -= (equation(x_deriv_val) / self.compute_derivative(x_deriv_val, equation))
        return x_deriv_val
            
        
    def solution(self, N = 10 ** 4):
        step_size = 5 / N
        x, t = self.x_initial, self.t_initial
        t_vals, x_vals = [t], [x]
        for _ in range(N):
            k1 = self.diffeq(self.newton_raphson(x,t), x, t)
            k2 = self.newton_raphson(x + 0.5 * step_size * k1, t + 0.5 * step_size)
            k3 = self.newton_raphson(x + 0.5 * step_size * k2, t + 0.5 * step_size)
            k4 = self.diffeq(self.newton_raphson(x + step_size * k3, t + step_size), x + step_size * k3, t + step_size)
            x += (step_size / 6) * (k1 + 2*k2 + 2*k3 + k4)
            t += step_size
            t_vals.append(t)
            x_vals.append(x)
        return t_vals, x_vals
    
    
    def interpolate(self, x_vals, t_vals):
        step_size = 5 / (10 ** 4)
        
        indices = [int(i / step_size) for i in [0,1,2,3,4,5]]
        t_samples = [t_vals[index] for index in indices]
        x_samples = [x_vals[index] for index in indices]
        matrix = [[0 for _ in range(6)] for _ in range(6)]
        for i in range(6):
            t_val = t_samples[i]
            for j in range(6):
                matrix[i][j] = np.pow(t_val, 5 - i)
        matrix = np.array(matrix)
        b = np.array([x for x in x_samples])
        coefficients = np.linalg.matmul(np.linalg.inv(matrix), b)
        print(coefficients)
    
    def plot_solution(self, t_vals, x_vals):
        plt.plot(t_vals, x_vals)
        plt.show()
        
class Q4:
    
    def __init__(self):
        self.eq1 = lambda v: v
        self.eq2: lambda x, y: -(x * y)
        self.boundary_conditions = [[0, 1], [2, 2]]
        
    def shoot(self, s):
        h = 0.01
        x, y = self.boundary_conditions[0]
        v = s
        x_vals, y_vals = [x], [y]
        x_stop, y_stop = self.boundary_conditions[1]
        while x <= x_stop:
            x_old, y_old = x,y
            y += (h * v)
            v += (h * -(x_old * y_old))
            x_vals.append(x)
            y_vals.append(y)
            x += h
        return x_vals, y_vals
        
    def shooting_residual(self, s):
      _, y_vals = self.shoot(s)
      y_stop_target = self.boundary_conditions[1][1]
      return y_vals[-1] - y_stop_target
    
    def compute_derivative(self, s):
        return (self.shooting_residual(s + 0.01) - self.shooting_residual(s - 0.01)) / 0.02
         
        
    def solve(self):
        s = 2
        while True:
            s -= self.shooting_residual(s) / self.compute_derivative(s)
            if abs(self.shooting_residual(s)) < 0.01:
                x_vals, y_vals = self.shoot(s)
                for i in range(len(x_vals)):
                    print(f"x value: {x_vals[i]}, y value: {y_vals[i]}")
                break
                

if __name__ == "__main__":
    Q1 = Q1()
    Q1.solution1()
    Q1.solution2()
    #Q3 = Q3()
    #t_vals, x_vals =  Q3.solution()
    #Q3.plot_solution(t_vals, x_vals)
    #Q3.interpolate(t_vals, x_vals)
    #Q4 = Q4()
    #Q4.solve()
    