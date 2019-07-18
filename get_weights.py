import tensorflow as tf

model_path = 'dae_model97'

tf.reset_default_graph()

with tf.Session() as sess:
        with tf.get_default_graph().as_default() as graph:
                saver = tf.train.import_meta_graph(model_path + '.meta')
                saver.restore(sess, model_path)

                ops = graph.get_operations()
                weights = graph.get_tensor_by_name('enc-w:0')

                print (ops)
                print (weights)



        # name_scope = 'encoder'
        # encoded_tensor = self.get_model_activation_func(name_scope, sess.graph)
        # input = graph.get_tensor_by_name('x-corr-input:0')

        # encoded_data = encoded_tensor.eval({input: data})
