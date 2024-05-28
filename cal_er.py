from betaincder import betainc
from scipy.stats import binom
from scipy.special import comb, perm
import matplotlib.pyplot as plt

def cal_upper_bound(delta, rho, k, gamma, num):
    result = 0
    for j in range(1, k + 1):
        for m in range(gamma, num):
            eta = betainc(rho, k - j + 1, j)
            comb_c = comb(num-1, m)
            binom_delta = binom.pmf(p=delta, n=k, k=j)
            result += comb_c *  pow(eta, m) * pow(1-eta, num-1-m) * binom_delta
    return result

def plot_upper_bound(y1, y2, y3, y4):

    plt.style.use('seaborn-v0_8')
    x = list(range(len(y1)))
    # figure1c
    # plt.plot(x, y1, color='#EFBB24', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.5$ $\rho_{k}=0.3$')
    # plt.plot(x, y2, color='#616138', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.4$ $\rho_{k}=0.4$')
    # plt.plot(x, y3, color='#77428D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.3$ $\rho_{k}=0.5$')
    # plt.plot(x, y4, color='#DB4D6D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.2$ $\rho_{k}=0.6$')
    # figure1d
    # plt.plot(x, y1, color='#EFBB24', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\rho_{k}=0.3$')
    # plt.plot(x, y2, color='#616138', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\rho_{k}=0.4$')
    # plt.plot(x, y3, color='#77428D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\rho_{k}=0.5$')
    # plt.plot(x, y4, color='#DB4D6D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\rho_{k}=0.6$')

    # # figure1 a/b
    plt.plot(x, y1, color='#EFBB24', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.2$')
    plt.plot(x, y2, color='#616138', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.4$')
    plt.plot(x, y3, color='#77428D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.6$')
    plt.plot(x, y4, color='#DB4D6D', linestyle='-', marker='o',markersize=12,linewidth=3,label=r'$\delta_{k}=0.8$')

    #plt.yticks(ticks=[0.05, 0.15, 0.25, 0.35],labels=['.05', '.15', '.25', '.35'],size=25, fontweight='bold') # figure 1 a
    #plt.yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0],size=25, fontweight='bold') # figure 1 b
    #plt.yticks(ticks=[0.0, 0.02, 0.04, 0.06, 0.08],labels=['.00', '.02', '.04', '.06', '.08'],size=25, fontweight='bold') # figure 1 c
    #plt.xticks(ticks=[0,1,2,3,4],labels=['1','2','3','4','5'],size=25, fontweight='bold')# Figure 1 c/d
    plt.xticks(ticks=[0,1,2,3,4,5],labels=['5','20','50','100','150','200'],size=25, fontweight='bold') # figure 1 a/b
    plt.legend(prop={'size': 25, 'weight': 'bold'})
    plt.show()
    return


k_list = [5, 20, 50, 100, 150, 200]
gamma_list = [1, 2, 3, 4, 5]
num = 3

list1=[]
list2=[]
list3=[]
list4=[]

# figure1 c
# for i in gamma_list:
#     list1.append(cal_upper_bound(0.5, 0.3, 50, num-i, num))
#     list2.append(cal_upper_bound(0.4, 0.4, 50, num-i, num))
#     list3.append(cal_upper_bound(0.3, 0.5, 50, num-i, num))
#     list4.append(cal_upper_bound(0.2, 0.6, 50, num-i, num))

# figure1 d
# for i in gamma_list:
#     list1.append(cal_upper_bound(0.3, 0.3, 20, num-i, num))
#     list2.append(cal_upper_bound(0.3, 0.4, 20, num-i, num))
#     list3.append(cal_upper_bound(0.3, 0.5, 20, num-i, num))
#     list4.append(cal_upper_bound(0.3, 0.6, 20, num-i, num))


# #figure1 b
# for i in k_list:
#     list1.append(cal_upper_bound(0.2, 0.8, i, num-1, num))
#     list2.append(cal_upper_bound(0.4, 0.8, i, num-1, num))
#     list3.append(cal_upper_bound(0.6, 0.8, i, num-1, num))
#     list4.append(cal_upper_bound(0.8, 0.8, i, num-1, num))

# #figure1 a
for i in k_list:
    list1.append(cal_upper_bound(0.2, 0.8, i, num-1, num))
    list2.append(cal_upper_bound(0.4, 0.8, i, num-1, num))
    list3.append(cal_upper_bound(0.6, 0.8, i, num-1, num))
    list4.append(cal_upper_bound(0.8, 0.8, i, num-1, num))



plot_upper_bound(list1, list2, list3, list4)


















