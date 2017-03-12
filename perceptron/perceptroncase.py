import tensorflow as tf
    
def main():
    num_nodes = 2
    output_units = 1
    w = tf.Variable(tf.truncated_normal([num_nodes,output_units], -0.1, 0.1))
    b = tf.Variable(tf.truncated_normal([output_units],0.1))
    x = tf.placeholder(tf.float32, shape = [None, num_nodes])
    y = tf.placeholder(tf.float32, shape = [None,output_units])
    
    output = tf.sigmoid(tf.matmul(x,w) + b)
    cross_entropy = tf.reduce_mean(tf.square(output - y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_x = [[1.0,1.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]]
    train_y = [[1.0],[0.0],[0.],[0.]]
   
    for i in range(1000):
        sess.run([train_step], feed_dict={x:train_x,y:train_y})

    test_x = [[0.0,1.0],[0.0,0.0],[1.0,1.0],[1.0,0.0]]
    print(sess.run(output, feed_dict={x:test_x}))

if __name__ == '__main__':
    main()   

    

    
    
    
    