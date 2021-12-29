def graph_svm(X,y,clf,codes,alpha):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    
    h = 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    ZZ=np.array([codes[i] for i in Z])
    ZZ=ZZ.reshape(xx.shape)

    cat_encoder = LabelEncoder()
    ycat=cat_encoder.fit_transform(y)

    plt.contourf(xx, yy, ZZ, cmap=plt.cm.coolwarm, alpha=alpha)

    plt.scatter(X[:, 0], X[:, 1], c=ycat, cmap=plt.cm.coolwarm)
    plt.xlabel('Formante 1 [Hz]')
    plt.ylabel('Formante 2 [Hz]')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xticks(())
    #plt.yticks(())
    plt.show()