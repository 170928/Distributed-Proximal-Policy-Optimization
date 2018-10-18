import globals as g


def UPDATE_EVENT_set():
    g.UPDATE_EVENT.set()

def UPDATE_EVENT_clear():
    g.UPDATE_EVENT.clear()

def UPDATE_EVENT_wait():
    g.UPDATE_EVENT.wait()



def ROLLING_EVENT_set():
    g.ROLLING_EVENT.set()

def ROLLING_EVENT_clear():
    g.ROLLING_EVENT.clear()

def ROLLING_EVENT_wait():
    g.ROLLING_EVENT.wait()

def ROLLING_EVENT_IS_SET():
    return g.ROLLING_EVENT.is_set()

def GLOBAL_UPDATE_COUNTER_inc():
    g.GLOBAL_UPDATE_COUNTER+=1

def GLOBAL_UPDATE_COUNTER_reset():
    g.GLOBAL_UPDATE_COUNTER=0

def GLOBAL_UPDATE_COUNTER_get():
    return g.GLOBAL_UPDATE_COUNTER

def GLOBAL_EP_inc():
    g.GLOBAL_EP+=1

def GLOBAL_EP_reset():
    g.GLOBAL_EP=0

def GLOBAL_EP_get():
    return g.GLOBAL_EP

def QUEUE_SIZE():
    return g.QUEUE.qsize()

def QUEUE_get():
    return g.QUEUE.get()

def QUEUE_PUT(temp):
    g.QUEUE.put(temp)

def GLOBAL_RUNNING_R_get():
    return g.GLOBAL_RUNNING_R

def GLOBAL_RUNNING_R_append(temp):
    g.GLOBAL_RUNNING_R.append(temp)