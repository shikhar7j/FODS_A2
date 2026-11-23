import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
import p1 as p1

y_train = p1.y_train
y_test = p1.y_test
Z_train,Z_test,mu,sd = p1.standardise(p1.X_train,p1.X_test)
S,lam,U = p1.pca(Z_train)
names = p1.feature_names
ks = [1,2,3,5,8]
folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)


def make_model():
    return LogisticRegression(C=1.0,max_iter=1000)


def test_accuracy(F_train,F_test):
    model = make_model().fit(F_train,y_train)
    return model.score(F_test,y_test)


def f_scores(Z,y):
    N,D = Z.shape
    C = 3
    overall = Z.mean(axis=0)
    between = np.zeros(D)
    within = np.zeros(D)
    for c in range(C):
        Zc = Z[y==c]
        mc = Zc.mean(axis=0)
        between += len(Zc)*(mc-overall)**2
        within += np.sum((Zc-mc)**2,axis=0)
    return (between/(C-1))/(within/(N-C))


def forward_selection(Z,y,k_max):
    chosen = []
    for step in range(k_max):
        best_j = None
        best_score = -np.inf
        for j in range(Z.shape[1]):
            if j in chosen:
                continue
            cols = chosen+[j]
            score = cross_val_score(make_model(),Z[:,cols],y,cv=folds,scoring='accuracy').mean()
            if score > best_score:
                best_score = score
                best_j = j
        chosen.append(best_j)
    return chosen


class analysis:

    def baseline():
        acc = test_accuracy(Z_train,Z_test)
        print(f"Baseline test accuracy: {acc:.4f}")
        return acc

    def run():
        F = f_scores(Z_train,y_train)
        rank = np.argsort(-F)
        sfs_order = forward_selection(Z_train,y_train,8)
        acc_pca = []
        acc_filter = []
        acc_wrap = []
        for k in ks:
            acc_pca.append(test_accuracy(Z_train@U[:,:k],Z_test@U[:,:k]))
            sel_f = list(rank[:k])
            acc_filter.append(test_accuracy(Z_train[:,sel_f],Z_test[:,sel_f]))
            sel_w = sfs_order[:k]
            acc_wrap.append(test_accuracy(Z_train[:,sel_w],Z_test[:,sel_w]))

        print(f"{'k':>3s}{'PCA':>10s}{'Filter':>10s}{'Wrapper':>10s}")
        for i,k in enumerate(ks):
            print(f"{k:3d}{acc_pca[i]:10.4f}{acc_filter[i]:10.4f}{acc_wrap[i]:10.4f}")

        print("F-score ranking (best first):")
        for j in rank:
            print(f"{names[j]}: {F[j]:.4f}")

        for k in [3,8]:
            sel_f = set(rank[:k])
            sel_w = set(sfs_order[:k])
            print(f"k = {k}")
            print("Filter:",[names[j] for j in rank[:k]])
            print("Wrapper:",[names[j] for j in sfs_order[:k]])
            print("Shared:",len(sel_f & sel_w))

        return np.array(acc_pca),np.array(acc_filter),np.array(acc_wrap)

    def check_sfs():
        sfs_order = forward_selection(Z_train,y_train,8)
        for k in [3,8]:
            sfs = SequentialFeatureSelector(make_model(),n_features_to_select=k,direction="forward",
                                            scoring="accuracy",cv=folds).fit(Z_train,y_train)
            lib = set(np.where(sfs.get_support())[0])
            print(f"k = {k} own SFS matches sklearn:",lib == set(sfs_order[:k]))

    def within_one(baseline,acc_pca,acc_filter,acc_wrap):
        threshold = baseline-1/54
        for name,acc in [('PCA',acc_pca),('Filter',acc_filter),('Wrapper',acc_wrap)]:
            hit = [k for k,a in zip(ks,acc) if a >= threshold-1e-12]
            print(name,"smallest k within one sample of baseline:",hit[0] if hit else None)

    def paragraph():
        p = (
            "The baseline test accuracy on all 13 features is 0.9815 (53/54), giving a tolerance threshold "
            "of >= 0.9815 - 1/54 = 0.9630 (52/54); the smallest k reaching this is k=3 for PCA (0.9815), "
            "k=5 for Filter (0.9815), and k=8 for Wrapper (0.9630). Although PCA never sees class labels "
            "while Filter and Wrapper do, PCA unexpectedly achieves the highest test accuracy across most k "
            "(reaching 1.0000 at k=5). This occurs because the principal axes of maximum variance in the Wine "
            "dataset naturally align with between-class separation, whereas greedy forward selection risks "
            "overfitting the training folds. However, with only 54 test samples, a difference of 1 test error "
            "represents an accuracy step of ~1.85%, meaning differences between the methods fall within normal "
            "sampling variability and do not permit high confidence in their relative ranking."
        )
        print("\nP4 Paragraph Deliverable:\n" + p)


class plotting:

    def accuracy_plot(baseline,acc_pca,acc_filter,acc_wrap, save_path="p4_accuracy.png"):
        fig,ax = plt.subplots(1,1,figsize=(8,5))
        ax.plot(ks,acc_pca,'o-',label='PCA')
        ax.plot(ks,acc_filter,'s-',label='Filter (F-score)')
        ax.plot(ks,acc_wrap,'^-',label='Wrapper (SFS)')
        ax.axhline(y=baseline,linestyle='--',color='k',label='Baseline (13 features)')
        ax.set(xlabel='k',ylabel='Test accuracy',title='Test Accuracy vs Number of Features')
        ax.set_xticks(ks)
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


if __name__ == "__main__":
    baseline = analysis.baseline()
    acc_pca,acc_filter,acc_wrap = analysis.run()
    analysis.check_sfs()
    analysis.within_one(baseline,acc_pca,acc_filter,acc_wrap)
    analysis.paragraph()
    plotting.accuracy_plot(baseline,acc_pca,acc_filter,acc_wrap)
