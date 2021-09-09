import GPUtil
import subprocess
from time import time, sleep

interval = 60
min_total = 6000 * 8
min_gpu = 6000
script = 'train/adapt_batch.sh'

# all GPUs
gpus = GPUtil.getGPUs()


def check_gpus():
    available = 0

    for gpu in gpus:
        if gpu.memoryFree < min_gpu:
            return
        available += gpu.memoryFree

    if available > min_total:
        free = True
        print('ENOUGH FREE GPU MEMORY FOUND --- CALLING SCRIPT!')
        subprocess.call(script, shell=True)

    print(available, 'available', end='\r')


# check if free
free = False
while not free:
    print('checking ...', end='\r')
    check_gpus()
    print('waiting ...', end='\r')
    sleep(interval - time() % interval)
