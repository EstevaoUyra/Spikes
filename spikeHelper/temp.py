def similarityMatrix(data,method='greek',compare=None):
    if compare == None:
        X = getX(data); W = X;
        y = data['y']; z = y;
    else:
        types = np.unique(data[compare].values)
        assert types.shape == (2,)
        X = getX(data[data[compare]==types[0]]);
        W = getX(data[data[compare]==types[1]]);
        y = data[data[compare]==types[0]]['y']
        z = data[data[compare]==types[1]]['y']

    if method == 'mah':
        empCov = EmpiricalCovariance()
        precision = empCov.fit(getX(data)).get_precision()
        dist = lambda u,v : mahalanobis(u,v,precision)
    elif method == 'greek':
        dist = euclidean

    times = np.unique(y)
    meanActivity = [ W[z==ti].mean(axis=0) for ti in np.unique(z)]
    distances = []
    for ti in times:
        distances.append([])
        distances[-1] = np.array([np.array([dist(u,v) for u in X[y==ti]]).mean() for v in meanActivity])
    sim = 1/np.array(distances)
    sns.heatmap(normRows(sim))
    plt.plot(sim.argmax(axis=1)+.5, np.arange(20), 'k')
    plt.title('Similarity between activity')
    if compare is not None:
        plt.xlabel(types[0])
        plt.ylabel(types[1])

    return sim
