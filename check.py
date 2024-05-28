import torch

dataset='cifar10'
pr=0.4
ir=0.0
tau=0.6
k=150

test1 = torch.load('/home/hs/partial_label_learning/testCLSP/saved/'+dataset+'/pr='+str(pr)+'_ir='+str(ir)+'_tau='+str(tau)+'_k='+str(k)+'_model=blip2_seed=7438.pt')
test2 = torch.load('/home/hs/partial_label_learning/clp/saved/'+dataset+'/pr='+str(pr)+'_nr=0.0_ir='+str(ir)+'_tau='+str(tau)+'_k='+str(k)+'_model=blip2_seed=7438.pt')

temp = (test1 -test2).sum()
# 216 353 371 49904

print(temp)
if temp==0 :
    print("1111111")
else:
    print("0000000")