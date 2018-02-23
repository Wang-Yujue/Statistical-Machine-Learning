import numpy as np

p_fail = 1/3
n_trails = 1000
n_test = 4

results = np.random.binomial(n_test, p_fail, n_trails)

p_work = sum(results == 0) / n_trails

p_A_fail = sum(results == 1) / (4 * sum(results != 0))

print("Probability to work correctly:{0}\t\t"
      "Probability that only A failed given "
      "that S is not operating properly:{1}\t".format(p_work,p_A_fail))
p1_hand = 16/81
p2_hand = 8/65
print("The results calculated by hand:\t --first:{0}\t --second;{1}".format(p1_hand,p2_hand))