import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
import argparse


def initialize():
    global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, UPDATE_EVENT, ROLLING_EVENT, QUEUE
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []


    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()
    '''
    Event 객체는 하나의 플래그와 set(), clear(), wait(), isSet() 의 4개의 메소드를 가지고 있는 객체이다.
    - 플래그 초기값 : 0
    - set()   :  플래그를 1로 설정
    - clear() : 플래그를 0으로 설정
    - wait()  : 플래그가 1이면 즉시 리턴, 0 이면 1 로 설정될때까지 대기
    - isSet() : 플래그 상태를 넘겨준다.
    활용법 : 두개의 쓰레드가 순서를 가지고 동작해야 할 필요가 있는 경우( 준비작업 -> 처리작업)
    '''

    QUEUE = queue.Queue()


