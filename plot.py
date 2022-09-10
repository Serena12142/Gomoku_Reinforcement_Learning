from matplotlib import pyplot as plt
import pickle
import numpy as np

losses=pickle.load(open("loss_data.pickle",'rb'))
plt.axis([0, len(losses['total'])+1,0, 10])
plt.yticks(np.arange(0, 10, step=0.5))
plt.plot([i for i in range(1,len(losses['total'])+1)], losses['total'],'r')
plt.plot([i for i in range(1,len(losses['total'])+1)], losses['policy'],'b')
plt.plot([i for i in range(1,len(losses['total'])+1)], losses['value'],'g')
plt.grid()
plt.show()

plt.axis([0, len(losses['total'])+1,0, 480])
plt.plot([i for i in range(1,len(losses['total'])+1)], losses['round'],'r.')
plt.grid()
plt.show()
