from learning_iil.demos.movidius.sim2real import model
import tensorflow as tf

x = tf.placeholder('float', [None, 60, 80, 3], name='input')
#
nn_model = model(x)
saver = tf.train.Saver(tf.global_variables())
#
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
#
checkpoint = tf.train.latest_checkpoint('../trained_models/alg_upms/3/ror_64_32_adag/')
#
saver.restore(session, checkpoint)
saver.save(session, './ncs_port')



