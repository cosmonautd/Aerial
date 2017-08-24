import gtde

frame = gtde.loadimage('aerial2.jpg')
estimator = gtde.GroundTraversalDifficultyEstimator(binary=True, granularity=128, threshold=50)
diffimage = estimator.computeimage(frame)
gtde.show2image(frame, diffimage)