#-*-coding:utf-8 -*-

import numpy as np 
import tensorflow as tf 
import os 

import json
import pymysql

#N*time 帧数* 一帧的维度
dims =63
time_steps = 60

batch_size = 16


def batch_generator(x, y, batch_size=batch_size): 
    offset = 0
    while True:
        offset += batch_size
        
        if offset == batch_size or offset >= len(x):
            [x, y] = np.random.shuffle([x, y])
            offset = batch_size
            
        X_batch = x[offset - batch_size: offset]    
        Y_batch = y[offset - batch_size: offset]
        
        yield (X_batch, Y_batch)

def batch_generator2(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

def save_one_class_data(label,data,num):
    label_0 = np.where(leap_label ==num)[0]
   
    data_0 =[]

    for i in label_0:
        d_ = leap_data[i]
        data_0.append(d_)

    np.savez("data_0.npz",data_0)

    D = np.load("data_0.npz")
    print(len(D))
    A = D["arr_0"]
    print(len(A))


conn = pymysql.connect(host = "218.68.6.114",user = "leapmotion",passwd = 'leapmotion',db = 'leapmotion_data',port =3306)
cur = conn.cursor()

cur.execute('select * from LeapMachine where id <=100')

'''
'(('{json}',id))'
'''
data_all = cur.fetchall()


#二维元组转 二维list
data_all = list(data_all)
for c in data_all:
    data_all[data_all.index(c)] = list(c)

print ("dataset shape:{}".format(np.shape(data_all)))

#取数据
'''
['{}','{}'],用于json.loads  then,字典去数据和label
'''
data_ = [ i[0] for i in  data_all]

leap_data =  []
leap_label =  []

for j in data_:
    leap_ = json.loads(j)
    label_ = leap_['ControllerTypeInt']
    data_ = leap_['leapdata']

    leap_data.append(data_)
    leap_label.append(label_)

#转numpy数组,方便后续操作
leap_data = np.array(leap_data,dtype= np.float32)
leap_label = np.array (leap_label)


#把每个动作的样本分开存储
#save_one_class_data(leap_label,leap_data,0)
#定义测试数据  和训练数据 ？？？


#打乱 借助index
index = [ i for i in range(len (leap_data)) ]
np.random.shuffle(index)

leap_data = leap_data[index]
leap_label = leap_label[index]

#one-hot label  N*2
depth =2
leap_label_oh = tf.one_hot(leap_label,depth,on_value=1.0)  


#批量函数，
#hidden lyers
lstm_size = 128
num_layers = 3

keep_prob = 0.5
n_classes = 2

learning_rate = 0.01
training_iters = 1000
max_steps = 100000
display_step = 200



lstm_inputs = tf.placeholder(dtype = np.float32,shape=(batch_size,time_steps,dims))
y = tf.placeholder("float", [None, n_classes])

weight = tf.Variable(tf.random_normal([lstm_size, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))


def get_a_cell(lstm_size,keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units = lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob)

    return drop


with tf.name_scope('lstm'):

    # cell = rnn_cell.LSTMCell(n_hidden,state_is_tuple = True)
    # cell = rnn_cell.MultiRNNCell([cell] * 2)

    cell = tf.nn.rnn_cell.MultiRNNCell( [get_a_cell(lstm_size,keep_prob) for _ in range(num_layers)])

    #h0
    initial_state =  cell.zero_state(batch_size,tf.float32)

    #batch* 
    lstm_outputs,final_state = tf.nn.dynamic_rnn(cell=cell,inputs = lstm_inputs,initial_state=initial_state)
    # 通过lstm_outputs得到概率
    # seq_output = tf.concat(lstm_outputs,1)
    # x = tf.reshape(seq_output,[-1,lstm_size])

    # with tf.variable_scope('softmax'):

    output = tf.transpose(lstm_outputs,[1,0,2])
    # last = tf.gather(output,tf.int16 int(output.get_shape())[0] - 1 )

    last = tf.gather(output,output.get_shape()[0] - 1 )

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

# prediction = RNN(x, weight, bias)

# Define loss and optimizer
loss_f = -tf.reduce_sum(y * tf.log(prediction))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_f)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
save_path = './leap.tf.model/'

with tf.Session() as session:
    # tmp = sess.run(leap_label_oh)
    batchdata = batch_generator(leap_data,leap_label_oh,batch_size)
    # print(tmp)
    session.run(init)
    step =0
    for itr in range(training_iters):  
          
        offset = (itr * batch_size) % (leap_label_oh.shape[0] - batch_size)
        batch_x = leap_data[offset:(offset + batch_size), :, :]
        batch_y = leap_label_oh[offset:(offset + batch_size), :]
        


        _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
        step += 1

        if epoch % display_step == 0:
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(epoch) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))

        if (step % save_every_n == 0):
            saver.save(session, os.path.join(save_path, 'model'), global_step=step)
        if step >= max_steps:
            break



# with tf.Session() as session:
#     # tmp = sess.run(leap_label_oh)
#     batchdata = batch_generator(leap_data,leap_label_oh,batch_size)
#     # print(tmp)
#     session.run(init)
#     step =0
#     for itr in range(training_iters):    
#         # offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
#         # batch_x = tr_features[offset:(offset + batch_size), :, :]
#         # batch_y = tr_labels[offset:(offset + batch_size), :]
#         for batch_x,batch_y in  batchdata:


#             _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
#             step += 1

#             if epoch % display_step == 0:
#                 # Calculate batch accuracy
#                 acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#                 # Calculate batch loss
#                 loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
#                 print ("Iter " + str(epoch) + ", Minibatch Loss= " + \
#                     "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                     "{:.5f}".format(acc))

#             if (step % save_every_n == 0):
#                 saver.save(session, os.path.join(save_path, 'model'), global_step=step)
#             if step >= max_steps:
#                 break

    # print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))