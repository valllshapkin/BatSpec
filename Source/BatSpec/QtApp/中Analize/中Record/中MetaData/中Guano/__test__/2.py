from guano import GuanoFile
g = GuanoFile('/MainData/Repo/golonchenroppi/BatSpec/Tests/MYODAS_20230624_005134.wav')
for key, value in g.items():
    print ('%s: %s' % (key, value))