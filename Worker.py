import numpy as np
import gym
import global_utils as gu


class Worker(object):
    def __init__(self, wid, GLOBAL_PPO, args):
        self.wid = wid
        self.env = gym.make(args.GAME).unwrapped
        self.ppo = GLOBAL_PPO
        self.COORD = None
        self.args = args

    def set_COORD(self, COORD):
        self.COORD = COORD

    def work(self):


        while not self.COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            r = 0

            observations = []
            actions = []
            rewards = []
            vpreds = []


            for t in range(self.args.EP_LEN):
                '''
                GLOBAL PPO가 업데이트 중 이라면,
                clear history buffer, use new policy to collect data
                '''
                if not gu.ROLLING_EVENT_IS_SET():
                    gu.ROLLING_EVENT_wait()
                    observations, actions, rewards, vpreds = [], [], [], []

                '''
                a :: [None] 의 discrete action index에 대한 모음 
                '''
                s = np.stack([s]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                '''
                return  (1) a : [1]
                        (2) v : [1,1]
                '''
                a = self.ppo.action_estimate(s)
                v = self.ppo.value_estimate(s)


                a = np.asscalar(a)
                v = np.asscalar(v)


                observations.append(s)
                actions.append(a)
                rewards.append(r - 1)  # 0 for not down, -11 for down. Reward engineering
                vpreds.append(v)

                s_, r, done, _ = self.env.step(a)

                if done: r = -10




                s = s_
                ep_r += r

                gu.GLOBAL_UPDATE_COUNTER_inc()

                '''
                Episode 가 끝나거나 globla iteration 기준 업데이트를 해야할 때! 
                '''
                if t == self.args.EP_LEN - 1 or gu.GLOBAL_UPDATE_COUNTER_get() >= self.args.BATCH_SIZE or done:
                    if done:
                        '''
                        Episode 가 끝났을 때 s에서의  value 값은 0 이다. 
                        '''
                        v_s_ = 0
                    else:
                        '''
                        Next state 에서의 value 값을 critic 학습에 사용하므로 얻어온다. 
                        '''
                        s_ = np.stack([s_]).astype(dtype=np.float32)
                        v_s_ = self.ppo.value_estimate(s_)

                    discounted_r = []  # compute discounted reward

                    for r in rewards[::-1]:
                        v_s_ = r + self.args.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()


                    '''
                    queue에 데이터를 넣기 위한 처리 및 queue에 저장
                    '''
                    bs, ba, br = np.vstack(observations), np.vstack(actions), np.vstack(np.array(discounted_r))
                    observations, actions, rewards, vpreds = [], [], [], []

                    gu.QUEUE_PUT(np.hstack((bs, ba, br)))


                    if gu.GLOBAL_UPDATE_COUNTER_get() >= self.args.BATCH_SIZE:
                        '''
                        Rolling event stop :: 워커를 통한 히스토리 수집 종료
                        글로벌 업데이트 
                        '''
                        gu.ROLLING_EVENT_clear()
                        gu.UPDATE_EVENT_set()

                    if gu.GLOBAL_EP_get() >= self.args.MAX_EP:  # stop training
                        self.COORD.request_stop()
                        break

                    if done: break

            # record reward changes, plot later
            if len(gu.GLOBAL_RUNNING_R_get()) == 0:
                gu.GLOBAL_RUNNING_R_append(ep_r)
            else:
                gu.GLOBAL_RUNNING_R_append(gu.GLOBAL_RUNNING_R_get()[-1] * 0.9 + ep_r * 0.1)
            gu.GLOBAL_EP_inc()
            print('{0:.1f}%'.format(gu.GLOBAL_EP_get() / self.args.MAX_EP * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )
