import tensorflow as tf

model_path = ''

tf.reset_default_graph()

with tf.Session() as sess:
        with tf.get_default_graph().as_default() as graph:
                saver = tf.train.import_meta_graph(model_path + '.meta')
                saver.restore(sess, model_path)

        # name_scope = 'encoder'
        # encoded_tensor = self.get_model_activation_func(name_scope, sess.graph)
        # input = graph.get_tensor_by_name('x-corr-input:0')

        # encoded_data = encoded_tensor.eval({input: data})
