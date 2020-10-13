
import tensorflow.compat.v1 as tf


tf.compat.v1.disable_eager_execution() 
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name="../efficientnet-l2-nasfpn-ssl/model.ckpt", tensor_name='', all_tensors=False)

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("../efficientnet-l2-nasfpn-ssl/model.ckpt.meta")
  new_saver.restore(sess, tf.train.latest_checkpoint('../efficientnet-l2-nasfpn-ssl/'))
  #saver = tf.train.Saver()
  #saver.restore(sess, "efficientnet-l2-nasfpn-ssl/model.ckpt")
  #for n in tf.get_default_graph().as_graph_def().node:
  #  print(n)
    
  writer = tf.summary.FileWriter("../efficientnet-l2-nasfpn-ssl/buildmodel_arch", sess.graph)
  writer.close()