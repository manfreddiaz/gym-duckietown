import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph



        # We restore the weights
# saver.restore(session, checkpoint)
session = tf.Session()
saver = tf.train.import_meta_graph('ncs_port' + '.meta', clear_devices=True)
checkpoint = tf.train.latest_checkpoint('.')
saver.restore(session, checkpoint)

output_graph_def = tf.graph_util.convert_variables_to_constants(
            session, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            ['dense_2/BiasAdd'] # The output node names are used to select the usefull nodes
        )


with tf.gfile.GFile('ncs_port.pa', "wb") as f:
    f.write(output_graph_def.SerializeToString())


freeze_graph(
    input_graph='ncs_port.pa',
    input_saver='',
    input_binary=True,
    input_checkpoint='ncs_port',
    output_graph='ncs_port.pb',
    output_node_names='dense_2/BiasAdd',
    restore_op_name='save/restore_all',
    clear_devices=True,
    initializer_nodes="",
    variable_names_blacklist='',
    filename_tensor_name=None
)