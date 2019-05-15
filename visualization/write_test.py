sample_input_path = '../app_specific_files/dummy_app/sample_input/'
N = 50
for i in range(0,N):
	file_name = '%s%dbotnet.ipsum'%(sample_input_path,i+1)
	with open(file_name,'+w') as file:
		line='This is sample test %d!'%(i+1)
		file.write(line)