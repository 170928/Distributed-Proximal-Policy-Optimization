import tensorflow as tf
import numpy as np
import global_utils as gu


class PPONet(object):
    def __init__(self, s_dim, a_dim, args):
        self.state_dim = s_dim
        self.action_dim = a_dim
        self.sess = tf.Session()
        self.args = args
        self.COORD = None

        with tf.variable_scope('ph'):
            self.obs = tf.placeholder(tf.float32, [None] + list(self.state_dim), 'observations')
            self.discounted_r = tf.placeholder(tf.float32, [None], 'discounted_reward')
            self.actions = tf.placeholder(tf.int32, [None], 'actions')
            self.advantages = tf.placeholder(tf.float32, [None], 'advantages')


        with tf.variable_scope('build'):
            '''
            value :: [None, 1]
            probs :: [None, action_dim]
            '''
            self.value, self.value_params = self.build_critic('c')
            self.action_probs, self.actor_params = self.build_actor('new')
            self.old_action_probs, self.old_actor_params = self.build_actor('old')


        with tf.variable_scope('adv'):
            '''
            discounted_r = reward(t) + r * value(t+1)
            '''
            self.adv = self.value - self.discounted_r

        with tf.variable_scope('critic_loss'):
            self.c_loss = tf.reduce_mean(tf.square(self.adv))
            self.c_train_op = tf.train.AdamOptimizer(self.args.critic_lr).minimize(self.c_loss)


        with tf.variable_scope('actor_update'):
            self.actor_update = [a.assign(b) for a, b in zip(self.old_actor_params, self.actor_params)]

        with tf.variable_scope('surrogate'):
            '''
            self.action_probs / old_action_probs :: [None, action_dim]
            self.actions :: [None] 
            pi(old_pi)_probs :: [None]
            ratio :: [None]
            surrogate :: [None]
            We need to convert self.actions to probability of choosen actions :: [None]
            '''
            action_indices = tf.stack([tf.range(tf.shape(self.actions)[0], dtype=tf.int32), self.actions], axis=1)
            pi_probs = tf.gather_nd(params=self.action_probs, indices=action_indices)
            old_pi_probs = tf.gather_nd(params=self.old_action_probs, indices=action_indices)
            ratio = pi_probs/old_pi_probs
            surrogate = ratio * self.advantages

        with tf.variable_scope('actor_optimize'):
            '''
            PPO surrogate function calculation
            '''
            self.a_loss = -tf.reduce_mean(tf.minimum(
                surrogate,
                tf.clip_by_value(ratio, 1. - self.args.epsilon_surrogate, 1. + self.args.epsilon_surrogate) * self.advantages))

            self.a_train_op = tf.train.AdamOptimizer(self.args.actor_lr).minimize(self.a_loss)
            self.sess.run(tf.global_variables_initializer())





    def set_COORD(self, COORD):
        self.COORD = COORD

    def build_actor(self, name):
        '''
        :param name:
        :return: action_probs :: [None, actino_dim]
        '''

        with tf.variable_scope(name+'actor'):
            h1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
            h2 = tf.layers.dense(inputs=h1, units=20, activation=tf.tanh)
            h3 = tf.layers.dense(inputs=h2, units=self.action_dim, activation=tf.tanh)
            action_probs = tf.layers.dense(inputs=h3, units=self.action_dim, activation=tf.nn.softmax)
            actor_scope = tf.get_variable_scope().name
            actor_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor_scope)

        return action_probs, actor_params

    def build_critic(self, name):
        '''
        :param name:
        :return: estimated value :: [None,1] because tf.layers.dense(v2, """" 1 """" ) <<----[None, units=1]
        '''
        with tf.variable_scope(name+'critic'):
            v1 = tf.layers.dense(inputs=self.obs, units=40, activation=tf.tanh)
            v2 = tf.layers.dense(inputs=v1, units=30, activation=tf.tanh)
            value = tf.layers.dense(v2, 1, activation=None)
            value_scope = tf.get_variable_scope().name
            value_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=value_scope)

            value = tf.reshape(value, [-1])
        return value, value_params

    def value_estimate(self, obs):
        '''
        :param obs:
        :return: [None]
        '''
        return self.sess.run(self.value, feed_dict={self.obs : obs})

    def action_estimate(self, obs):
        '''
        :param obs: self.action_probs :: [None, action_dim]
        :return:  estimated_probs :: [None, action_dim]
                  actions (1) :: [None, 1]
                  actions (2) :: [None]

        '''
        actions = tf.multinomial(tf.log(self.action_probs), num_samples=1)
        actions = tf.reshape(actions, [-1])
        return self.sess.run(actions, feed_dict={self.obs : obs})

    def update(self):
        '''
        Cartpole 의 경우 env.observation_state.shape :: (4, )
                        evn.obseration_state.shape[0] :: 4
                        list(env.observation_state.shape) :: [4]
        '''
        S_DIM = self.state_dim[0]
        UPDATE_STEP = 15
        while not self.COORD.should_stop():
            if gu.GLOBAL_EP_get() < self.args.MAX_EP:
                '''
                GLOBAL UPDATE Stop
                충분한 batch size를 모을 때까지 대기 
                '''
                gu.UPDATE_EVENT_wait()

                '''
                Params update
                old actor params 를 current actor param 으로 업데이트 한다.
                '''
                self.sess.run(self.actor_update)

                '''
                모델의 업데이트를 위해서 다른 worker들이 만든 데이터를 queue로 부터 가져온다.
                '''
                data = [gu.QUEUE_get() for _ in range(gu.QUEUE_SIZE())]
                data = np.vstack(data)
                '''
                vstack 명령은 열의 수가 같은 두 개 이상의 배열을 위아래로 연결하여 행의 수가 더 많은 배열을 만든다. 
                연결할 배열은 마찬가지로 하나의 리스트에 담아서 보내야 한다.
                '''
                #print(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1].ravel(), data[:, -1:].ravel()
                '''
                s :: [None, S_dim]
                a :: [None]
                r :: [None]
                '''

                adv = self.sess.run(self.adv, {self.obs: s, self.discounted_r: r})

                '''
                actor & critic 을 업데이트 
                '''

                [self.sess.run(self.a_train_op, {self.obs: s, self.actions: a, self.advantages: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.c_train_op, {self.obs: s, self.discounted_r: r}) for _ in range(UPDATE_STEP)]

                '''
                Global variasbles control
                (1) update finish 
                (2) reset global update count
                (3) rollout available
                '''
                gu.UPDATE_EVENT_clear()  # updating finished
                gu.GLOBAL_UPDATE_COUNTER_reset()  # reset counter
                gu.ROLLING_EVENT_set() # set roll-out available
