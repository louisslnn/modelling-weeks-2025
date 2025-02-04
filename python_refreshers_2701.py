import matplotlib.pyplot as plt
import numpy as np

def plot_sines(epsilon):
    x = np.arange(-2, 2, epsilon)
    y1 = np.sin(2*np.pi*x)
    y2 = np.sin(np.pi*x)
    plt.plot(x,y1, 'b')
    plt.plot(x,y2, 'r:')
    plt.show()

def gaussian(a,b,c,x):
    return a * np.exp(-(x-b)**2/2*c**2)    

def plot_gaussian(a,b,c,inf,sup,n):
    x = np.linspace(inf, sup, n)
    y = gaussian(a, b, c, x)
    plt.plot(x, y)

mu_women = 164.7
sigma_women = 7.1
mu_men = 178.4
sigma_men = 7.6

a_women = 1 / sigma_women * np.sqrt(2*np.pi)

a_men = 1 / sigma_men * np.sqrt(2*np.pi)

plot_gaussian(a_women, mu_women, sigma_women, 140, 205, 200)
plot_gaussian(a_men, mu_men, sigma_men, 140, 205, 200)

heights = np.random.normal(mu_women, sigma_women, 1000)

def plot_women_heights():
    plt.hist(heights, bins=1000, density=True)
    plt.title("Height distribution")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

plot_women_heights()