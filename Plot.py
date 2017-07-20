import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]

#motor1 test result
motor1_test_loss = [0.3029, 0.2937, 0.2675, 0.2266, 0.1824, 0.1735]
motor1_test_acc = [0.9589, 0.9591, 0.9643, 0.9648, 0.9659, 0.9638]

#motor2 test result
motor2_test_loss = [0.2549, 0.2097, 0.1990, 0.1957, 0.1876, 0.1953]
motor2_test_acc = [0.9578, 0.9643, 0.9642, 0.9643, 0.9627, 0.9519]

#motor3 test result
motor3_test_loss = [0.2645, 0.2254, 0.1863, 0.1685, 0.1585, 0.1575]
motor3_test_acc = [0.9643, 0.9639, 0.9654, 0.9657, 0.9653, 0.9631]

with sns.axes_style('darkgrid'):
    plt.subplot(321)
    plt.plot(x, motor1_test_acc)
    plt.title('acc of motor1 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(322)
    plt.plot(x, motor1_test_loss)
    plt.title('loss of motor1 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')

    plt.subplot(323)
    plt.plot(x, motor2_test_acc)
    plt.title('acc of motor2 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(324)
    plt.plot(x, motor2_test_loss)
    plt.title('loss of motor2 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')

    plt.subplot(325)
    plt.plot(x, motor3_test_acc)
    plt.title('acc of motor3 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('acc')

    plt.subplot(326)
    plt.plot(x, motor3_test_loss)
    plt.title('loss of motor3 test')
    plt.xlabel('train_num_motors')
    plt.ylabel('loss')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.45)
    plt.show()

'''
sns.set(style="darkgrid", color_codes=True)
data = pd.DataFrame({'x': x, 'y': motor2_test_acc})
plt.subplot(211)
ax = sns.factorplot(x='x', y='y', truncate=True, size=5, data=data)
plt.subplot(212)
ax1 = sns.factorplot(x='x', y='y', truncate=True, size=5, data=data)
plt.show()
'''