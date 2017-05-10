def kflops(n):
	return 3*(n**2) - 3

with open('times') as f:
	times = f.read()

times = times.split('\n')

for l in times:
	if l == '':
		continue

	ls = l.split()
	n = int(ls[0]) + 1
	ts = (2*(n**2)) / (float(ls[1]) / 1000)
	tc = (2*(n**2)) / (float(ls[2]) / 1000)
	kc = kflops(n) / (float(ls[3]) / 1000)
	print('{} {} {} {}'.format(n - 1, ts / (10**6), tc / (10**6), kc / (10**6)))