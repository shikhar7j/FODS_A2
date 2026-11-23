import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import p1 as p1

X_train = p1.X_train
X_test = p1.X_test
y_train = p1.y_train
y_test = p1.y_test


def preprocess(variant):
    if variant == 'V1':
        mu = X_train.mean(axis=0)
        return X_train-mu,X_test-mu
    if variant == 'V2':
        mn = X_train.min(axis=0)
        mx = X_train.max(axis=0)
        tr = (X_train-mn)/(mx-mn)
        ts = (X_test-mn)/(mx-mn)
        mu = tr.mean(axis=0)
        return tr-mu,ts-mu
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0,ddof=1)
    return (X_train-mu)/sd,(X_test-mu)/sd


def run_variant(variant):
    P_train,P_test = preprocess(variant)
    S,lam,U = p1.pca(P_train)
    r = lam/np.sum(lam)
    C_train = P_train@U[:,:2]
    C_test = P_test@U[:,:2]
    knn = KNeighborsClassifier(n_neighbors=5).fit(C_train,y_train)
    acc = knn.score(C_test,y_test)
    top = p1.feature_names[int(np.argmax(np.abs(U[:,0])))]
    return r,top,acc,C_train,C_test


class analysis:

    def table():
        print(f"{'Variant':8s}{'r1':>9s}{'r2':>9s}{'r1+r2':>9s}  {'Top PC1 feature':32s}{'5-NN acc':>9s}")
        for v in ['V1','V2','V3']:
            r,top,acc,_,_ = run_variant(v)
            print(f"{v:8s}{r[0]:9.4f}{r[1]:9.4f}{r[0]+r[1]:9.4f}  {top:32s}{acc:9.4f}")

    def raw_variances():
        var = X_train.var(axis=0,ddof=1)
        top = np.argsort(-var)[:3]
        print("Three largest raw sample variances (V1):")
        for j in top:
            print(f"{p1.feature_names[j]}: {var[j]:.4f}")

    def paragraph():
        p = (
            "Variant V1 applies centering without scaling, causing the feature with the largest scale "
            "(proline, sample variance ~ 94906.71) to dominate the objective u^T S u and align PC1 almost entirely "
            "with proline, yielding r1 = 99.78%. In contrast, V3 standardises all features to unit variance, "
            "distributing variance across features so r1 is only 35.73%. While V1 captures 99.98% of total variance "
            "in two components versus 54.94% for V3, V1 yields a much lower 5-NN test accuracy (72.22% vs. 94.44%). "
            "This demonstrates that high total variance does not necessarily reflect discriminative class structure; "
            "large raw variance often reflects arbitrary measurement units rather than separability. Therefore, V3 "
            "(Z-score standardisation) should be used before PCA so features contribute based on correlation structure "
            "rather than measurement scale."
        )
        print("\nP3 Paragraph Deliverable:\n" + p)


class plotting:

    def projections(save_path="p3_projections.png"):
        colors = ['tab:blue','tab:orange','tab:green']
        fig,axes = plt.subplots(1,3,figsize=(18,5))
        for ax,v in zip(axes,['V1','V2','V3']):
            r,top,acc,C_train,C_test = run_variant(v)
            for c in range(3):
                ax.scatter(C_train[y_train==c,0],C_train[y_train==c,1],c=colors[c],marker='o',label=f'Train class {c}')
                ax.scatter(C_test[y_test==c,0],C_test[y_test==c,1],c=colors[c],marker='x',label=f'Test class {c}')
            ax.set(xlabel='PC1',ylabel='PC2',title=f'{v} PC1-PC2 projection')
            ax.grid()
            ax.legend(fontsize=7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


if __name__ == "__main__":
    analysis.table()
    analysis.raw_variances()
    analysis.paragraph()
    plotting.projections()
