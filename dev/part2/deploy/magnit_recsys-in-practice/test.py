import solution
import time


t1 = time.time()
solution.solution('../data/train_joke_df.csv', '../data/output_df.csv')
print('time:', time.time() - t1)