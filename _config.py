"""
aggragete configurations.
"""

import tensorflow as tf


###########  DATA  ####################
tf.app.flags.DEFINE_string("dir_train", "./data",
                           "train directory.")
tf.app.flags.DEFINE_string("dir_pretrain", None,
                           "pretrain directory.")

tf.app.flags.DEFINE_string("dir_val", "./data2",
                           "validation directory.")

tf.app.flags.DEFINE_string("path_targets", "./target_id.txt",
                           "file contains target category ids.")


tf.app.flags.DEFINE_integer('image_width', 224,
                            """all image resize to this width""")
tf.app.flags.DEFINE_integer('image_height', 224,
                            """all image resize to this height""")

tf.app.flags.DEFINE_integer('batch_size', 16,
                            """mini-batch size""")

############## LOG ######################
tf.app.flags.DEFINE_string('dir_log', './log',
                           """Directory where to write train logs """)
tf.app.flags.DEFINE_string('dir_parameter', './parameter',
                           """Directory where to write parameters""")
tf.app.flags.DEFINE_string('dir_log_val', './validation_log',
                           """Directory where to write validation logs """)
tf.app.flags.DEFINE_integer('eval_interval_secs', 120,
                            """How often to run the eval.""")



############ OPTIMIZE ###################
tf.app.flags.DEFINE_integer('max_steps', 500000,
                            """max_steps""")
tf.app.flags.DEFINE_float('lr', 0.01,
                            """initial learning rate.""")
tf.app.flags.DEFINE_float('decay_rate', 0.1,
                            """decay rate.""")
tf.app.flags.DEFINE_integer('decay_steps', 40000,
                            """decay_steps""")
