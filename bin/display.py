import derp.util
import time
while True:
    time.sleep(1E-3)
    print(derp.util.get_timestamp() / 1E9, end='\r')
