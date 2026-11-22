import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train_data = np.loadtxt('./data/testing_training/wine_train.csv',delimiter=',',skiprows=1)
test_data = np.loadtxt('./data/testing_training/wine_test.csv',delimiter=',',skiprows=1)

feature_names = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols',
                 'flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue',
                 'od280/od315_of_diluted_wines','proline']

X_train = train_data[:,:13]
y_train = train_data[:,13].astype(int)
X_test = test_data[:,:13]
y_test = test_data[:,13].astype(int)


def standardise(X_train,X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0,ddof=1)
    Z_train = (X_train-mu)/sd
    Z_test = (X_test-mu)/sd
    return Z_train,Z_test,mu,sd


def fix_sign(U):
    U = U.copy()
    for j in range(U.shape[1]):
        if U[np.argmax(np.abs(U[:,j])),j] < 0:
            U[:,j] = -U[:,j]
    return U


def covariance(Z):
    Zc = Z-Z.mean(axis=0)
    return (Zc.T@Zc)/(Z.shape[0]-1)


def pca(Z):
    S = covariance(Z)
    lam,U = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    U = fix_sign(U[:,order])
    return S,lam,U


class report:

    def eigenvalues(Z):
        S,lam,U = pca(Z)
        for j in range(len(lam)):
            print(f"lambda {j+1} = {lam[j]:.4f}")
        return S,lam,U

    def orthonormality(Z):
        S,lam,U = pca(Z)
        orth = np.linalg.norm(U.T@U-np.identity(U.shape[1]),'fro')
        trace_gap = abs(np.trace(S)-np.sum(lam))
        print("||U^T U - I||_F:",f"{orth:.1e}")
        print("|tr(S) - sum(lambda)|:",f"{trace_gap:.1e}")
        print(f"tr(S): {np.trace(S):.4f}")

    def verification(Z):
        S,lam,U = pca(Z)
        sk = PCA(n_components=13,svd_solver="full").fit(Z)
        sk_U = fix_sign(sk.components_.T)
        print("Max |lambda - explained_variance_|:",f"{np.max(np.abs(lam-sk.explained_variance_)):.1e}")
        print("Max |U - components_.T|:",f"{np.max(np.abs(U-sk_U)):.1e}")

    def loadings(Z):
        S,lam,U = pca(Z)
        print(f"{'Feature':32s}{'u1':>10s}{'u2':>10s}")
        for j in range(13):
            print(f"{feature_names[j]:32s}{U[j,0]:10.4f}{U[j,1]:10.4f}")
        for pc in range(2):
            top = np.argsort(-np.abs(U[:,pc]))[:3]
            print(f"Top 3 |loading| on PC{pc+1}:",[feature_names[j] for j in top])


if __name__ == "__main__":
    Z_train,Z_test,mu,sd = standardise(X_train,X_test)
    report.eigenvalues(Z_train)
    report.orthonormality(Z_train)
    report.verification(Z_train)
    report.loadings(Z_train)
