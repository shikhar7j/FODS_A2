import numpy as np
import matplotlib.pyplot as plt
import p1 as p1

Z_train,Z_test,mu,sd = p1.standardise(p1.X_train,p1.X_test)
S,lam,U = p1.pca(Z_train)
D = len(lam)


def variance_ratios(lam):
    r = lam/np.sum(lam)
    R = np.cumsum(r)
    return r,R


def recon_error(Z,U,k):
    Uk = U[:,:k]
    Z_hat = Z@Uk@Uk.T
    return np.sum((Z-Z_hat)**2)/(Z.shape[0]-1)


class analysis:

    def choose_k(lam):
        r,R = variance_ratios(lam)
        k_elbow = max(k for k in range(1,D+1) if r[k-1] >= 0.10)
        print("k_elbow:",k_elbow)
        print(f"R(k_elbow): {R[k_elbow-1]:.4f}")
        for tau in [0.80,0.90,0.95]:
            k_tau = int(np.argmax(R >= tau))+1
            print(f"k_{tau:.2f}:",k_tau)
        k_kaiser = int(np.sum(lam > np.trace(S)/D))
        print("Kaiser count:",k_kaiser)
        return k_elbow,k_kaiser

    def table(lam):
        r,R = variance_ratios(lam)
        print(f"{'k':>3s}{'r_k':>10s}{'R(k)':>10s}{'J_train(k)':>14s}{'J_test(k)':>14s}")
        J_tr = []
        J_ts = []
        for k in range(1,D+1):
            jtr = recon_error(Z_train,U,k)
            jts = recon_error(Z_test,U,k)
            J_tr.append(jtr)
            J_ts.append(jts)
            print(f"{k:3d}{r[k-1]:10.4f}{R[k-1]:10.4f}{jtr:14.4f}{jts:14.4f}")
        return np.array(J_tr),np.array(J_ts)

    def identity_check(lam):
        gaps = []
        for k in range(1,D+1):
            gaps.append(abs(recon_error(Z_train,U,k)-np.sum(lam[k:])))
        print("max_k |J_train(k) - sum_{j>k} lambda_j|:",f"{max(gaps):.1e}")

    def test_monotone(J_ts):
        print("J_test non-increasing:",bool(np.all(np.diff(J_ts) <= 1e-12)))


class plotting:

    def scree(lam):
        r,R = variance_ratios(lam)
        k_elbow = max(k for k in range(1,D+1) if r[k-1] >= 0.10)
        fig,ax = plt.subplots(1,1,figsize=(8,5))
        ax.bar(np.arange(1,D+1),r,label='Explained variance ratio')
        ax.axvline(x=k_elbow,linestyle='--',color='r',label=f'k_elbow = {k_elbow}')
        ax.set(xlabel='Component j',ylabel='r_j',title='Scree Plot')
        ax.set_xticks(np.arange(1,D+1))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

    def cumulative(lam):
        r,R = variance_ratios(lam)
        fig,ax = plt.subplots(1,1,figsize=(8,5))
        ax.plot(np.arange(1,D+1),R,'o-',label='R(k)')
        for tau in [0.80,0.90,0.95]:
            ax.axhline(y=tau,linestyle='--',label=f'{tau:.2f}',color={0.80:'g',0.90:'orange',0.95:'r'}[tau])
        ax.set(xlabel='k',ylabel='R(k)',title='Cumulative Explained Variance')
        ax.set_xticks(np.arange(1,D+1))
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

    def reconstruction(J_tr,J_ts):
        ks = np.arange(1,D+1)
        fig,ax = plt.subplots(1,1,figsize=(8,5))
        ax.plot(ks,J_tr,'o-',label='J_train(k)')
        ax.plot(ks,J_ts,'s-',label='J_test(k)')
        ax.set(xlabel='k',ylabel='Reconstruction error',title='Reconstruction Error vs k')
        ax.set_xticks(ks)
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analysis.choose_k(lam)
    J_tr,J_ts = analysis.table(lam)
    analysis.identity_check(lam)
    analysis.test_monotone(J_ts)
    plotting.scree(lam)
    plotting.cumulative(lam)
    plotting.reconstruction(J_tr,J_ts)
