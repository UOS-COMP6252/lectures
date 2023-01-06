import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("data-with-without-batch-norm.csv")
x=df.iloc[:,0].to_numpy()
t_acc_wo_bn=df['Valid-acc-with-BN'].to_numpy()
v_acc_wo_bn=df['Valid-acc-without-BN'].to_numpy()
plt.plot(x,t_acc_wo_bn,label='BN')
plt.plot(x,v_acc_wo_bn,label='No BN')
plt.xlim(0,50)
plt.ylim(0.5,0.8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Validation accuracy")
plt.show()
