from multiprocessing import Pool
import workers
if __name__ ==  '__main__': 
 num_processors = 3
 p=Pool(processes = num_processors)
 output = p.map(workers.worker,[i for i in range(0,3)])
 print(output)