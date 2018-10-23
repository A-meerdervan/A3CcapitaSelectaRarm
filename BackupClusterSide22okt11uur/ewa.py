import multiprocessing

num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
print(num_workers)
