#coding=utf-8
import numpy as np
import tensorflow as tf
import time
from basic_layer.NN_adam import NN
from util.Printer import TIPrint
from util.batcher.equal_len.batcher_p import batcher
from util.AccCalculater import cau_recall_mrr_org
from util.AccCalculater import cau_samples_recall_mrr
from util.Pooler import pooler
from basic_layer.FwNn3AttLayer import FwNnAttLayer
from util.FileDumpLoad import dump_file


class Seq2SeqAttNN(NN):
    """
    The memory network with context attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        super(Seq2SeqAttNN, self).__init__(config)
        self.config = None
        if config != None:
            self.config = config
            # the config.
            self.datas = config['dataset']
            self.nepoch = config['nepoch']  # the train epoches.
            self.batch_size = config['batch_size']  # the max train batch size.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            # the base of the initialization of the parameters.
            self.stddev = config['stddev']
            self.edim = config['edim']  # the dim of the embedding.
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            self.n_items = config["n_items"]
            # the pad id in the embedding dictionary.
            self.pad_idx = config['pad_idx']

            # the pre-train embedding.
            ## shape = [nwords, edim]
            self.pre_embedding = config['pre_embedding']

            # generate the pre_embedding mask.
            self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
            self.pre_embedding_mask[self.pad_idx] = 0

            # update the pre-train embedding or not.
            self.emb_up = config['emb_up']

            # the active function.
            self.active = config['active']

            # hidden size
            self.hidden_size = config['hidden_size']

            self.is_print = config['is_print']

            self.cut_off = config["cut_off"]
        self.is_first = True
        # the input.
        self.inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None


    def build_model(self):
        '''
        build the MemNN model
        '''
        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None,None],
            name="inputs"
        )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )


        batch_size = tf.shape(self.inputs)[0]

        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )

        self.lab_input = tf.placeholder(
            tf.int32,
            [None],
            name="lab_input"
        )

        # the lookup dict.
        self.embe_dict = tf.Variable(
            self.pre_embedding,
            dtype=tf.float32,
            trainable=self.emb_up
        )

        self.pe_mask = tf.Variable(
            self.pre_embedding_mask,
            dtype=tf.float32,
            trainable=False
        )
        self.embe_dict *= self.pe_mask

        sent_bitmap = tf.ones_like(tf.cast(self.inputs, tf.float32))

        inputs = tf.nn.embedding_lookup(self.embe_dict, self.inputs,max_norm=1)
        lastinputs= tf.nn.embedding_lookup(self.embe_dict, self.last_inputs,max_norm=1)

        org_memory = inputs


        pool_out = pooler(
            org_memory,
            'mean',
            axis=1,
            sequence_length = tf.cast(tf.reshape(self.sequence_length,[batch_size, 1]), tf.float32)
        )
        pool_out = tf.reshape(pool_out,[-1,self.hidden_size])

        attlayer = FwNnAttLayer(
            self.edim,
            active=self.active,
            stddev=self.stddev,
            norm_type='none'
        )
        attout, alph= attlayer.forward(org_memory,lastinputs,pool_out,sent_bitmap)
        attout = tf.reshape(attout,[-1,self.edim]) + pool_out
        self.alph = tf.reshape(alph,[batch_size,1,-1])

        self.w1 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )

        self.w2 = tf.Variable(
            tf.random_normal([self.edim, self.edim], stddev=self.stddev),
            trainable=True
        )
        attout = tf.tanh(tf.matmul(attout,self.w1))
        # attout = tf.nn.dropout(attout, self.output_keep_probs)
        lastinputs= tf.tanh(tf.matmul(lastinputs,self.w2))
        # lastinputs= tf.nn.dropout(lastinputs, self.output_keep_probs)
        prod = attout * lastinputs
        sco_mat = tf.matmul(prod,self.embe_dict[1:],transpose_b= True)
        self.softmax_input = sco_mat
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat,labels = self.lab_input)

        # the optimize.
        self.params = tf.trainable_variables()
        self.optimize = super(Seq2SeqAttNN, self).optimize_normal(
            self.loss, self.params)


    def train(self,sess, train_data, test_data=None,saver = None, threshold_acc=0.99):


        max_recall = 0.0
        max_mrr = 0.0
        max_train_acc = 0.0
        for epoch in range(self.nepoch):   # epoch round.
            batch = 0
            c = []
            cost = 0.0  # the cost of each epoch.
            bt = batcher(
                samples=train_data.samples,
                class_num= self.n_items,
                random=True
            )
            while bt.has_next():    # batch round.
                # get this batch data
                batch_data = bt.next_batch()
                # build the feed_dict
                # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
                batch_lenth = len(batch_data['in_idxes'])
                event = len(batch_data['in_idxes'][0])

                if batch_lenth > self.batch_size:
                    patch_len = int(batch_lenth / self.batch_size)
                    remain = int(batch_lenth % self.batch_size)
                    i = 0
                    for x in range(patch_len):
                        tmp_in_data = batch_data['in_idxes'][i:i+self.batch_size]
                        tmp_out_data = batch_data['out_idxes'][i:i+self.batch_size]
                        for s in range(len(tmp_in_data[0])):
                            batch_in = []
                            batch_out = []
                            batch_last = []
                            batch_seq_l = []
                            for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):

                               _in = tmp_in[s]
                               _out = tmp_out[s]-1
                               batch_last.append(_in)
                               batch_in.append(tmp_in[:s + 1])
                               batch_out.append(_out)
                               batch_seq_l.append(s + 1)
                            feed_dict = {
                                self.inputs: batch_in,
                                self.last_inputs: batch_last,
                                self.lab_input: batch_out,
                                self.sequence_length: batch_seq_l

                            }
                            # train
                            crt_loss, crt_step, opt, embe_dict = sess.run(
                                [self.loss, self.global_step, self.optimize, self.embe_dict],
                                feed_dict=feed_dict
                            )

                            # cost = np.mean(crt_loss)
                            c += list(crt_loss)
                            # print("Batch:" + str(batch) + ",cost:" + str(cost))
                            batch += 1
                        i += self.batch_size
                    if remain > 0:
                        # print (i, remain)
                        tmp_in_data = batch_data['in_idxes'][i:]
                        tmp_out_data = batch_data['out_idxes'][i:]
                        for s in range(len(tmp_in_data[0])):
                            batch_in = []
                            batch_out = []
                            batch_last = []
                            batch_seq_l = []
                            for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                                _in = tmp_in[s]
                                _out = tmp_out[s] - 1
                                batch_last.append(_in)
                                batch_in.append(tmp_in[:s + 1])
                                batch_out.append(_out)
                                batch_seq_l.append(s + 1)
                            feed_dict = {
                                self.inputs: batch_in,
                                self.last_inputs: batch_last,
                                self.lab_input: batch_out,
                                self.sequence_length: batch_seq_l

                            }
                            # train
                            crt_loss, crt_step, opt, embe_dict = sess.run(
                                [self.loss, self.global_step, self.optimize, self.embe_dict],
                                feed_dict=feed_dict
                            )

                            # cost = np.mean(crt_loss)
                            c += list(crt_loss)
                            # print("Batch:" + str(batch) + ",cost:" + str(cost))
                            batch += 1
                else:
                    tmp_in_data = batch_data['in_idxes']
                    tmp_out_data = batch_data['out_idxes']
                    for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            _in = tmp_in[s]
                            _out = tmp_out[s] - 1
                            batch_last.append(_in)
                            batch_in.append(tmp_in[:s + 1])
                            batch_out.append(_out)
                            batch_seq_l.append(s + 1)
                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l

                        }
                        # train
                        crt_loss, crt_step, opt, embe_dict = sess.run(
                            [self.loss, self.global_step, self.optimize, self.embe_dict],
                            feed_dict=feed_dict
                        )

                        # cost = np.mean(crt_loss)
                        c+= list(crt_loss)
                        # print("Batch:" + str(batch) + ",cost:" + str(cost))
                        batch += 1
            # train_acc = self.test(sess,train_data)
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))
            if test_data != None:
                recall, mrr = self.test(sess, test_data)
                print(recall, mrr)
                if max_recall < recall:
                    max_recall = recall
                    max_mrr = mrr
                    test_data.update_best()
                    if max_recall > threshold_acc:
                        self.save_model(sess, self.config, saver)
                print ("                   max_recall: " + str(max_recall)+" max_mrr: "+str(max_mrr))
                test_data.flush()
        if self.is_print:
            TIPrint(test_data.samples, self.config,
                    {'recall': max_recall, 'mrr': max_mrr}, True)

    def test(self,sess,test_data):

        # calculate the acc
        print('Measuring Recall@{} and MRR@{}'.format(self.cut_off, self.cut_off))

        mrr, recall = [], []
        c_loss =[]
        batch = 0
        bt = batcher(
            samples = test_data.samples,
            class_num = self.n_items,
            random = False
        )
        while bt.has_next():    # batch round.
            # get this batch data
            batch_data = bt.next_batch()
            # build the feed_dict
            # for x,y in zip(batch_data['in_idxes'],batch_data['out_idxes']):
            batch_lenth = len(batch_data['in_idxes'])
            event = len(batch_data['in_idxes'][0])
            if batch_lenth > self.batch_size:
                patch_len = int(batch_lenth / self.batch_size)
                remain = int(batch_lenth % self.batch_size)
                i = 0
                for x in range(patch_len):
                    tmp_in_data = batch_data['in_idxes'][i:i+self.batch_size]
                    tmp_out_data = batch_data['out_idxes'][i:i+self.batch_size]
                    tmp_batch_ids = batch_data['batch_ids'][i:i+self.batch_size]
                    for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            _in = tmp_in[s]
                            _out = tmp_out[s] - 1
                            batch_last.append(_in)
                            batch_in.append(tmp_in[:s + 1])
                            batch_out.append(_out)
                            batch_seq_l.append(s + 1)
                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l

                        }
                        # train
                        preds, loss, alpha = sess.run(
                            [self.softmax_input, self.loss, self.alph],
                            feed_dict=feed_dict
                        )
                        t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
                        test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                        test_data.pack_preds(ranks, tmp_batch_ids)
                        c_loss += list(loss)
                        recall += t_r
                        mrr += t_m
                        batch += 1
                    i += self.batch_size
                if remain > 0:
                    # print (i, remain)
                    tmp_in_data = batch_data['in_idxes'][i:]
                    tmp_out_data = batch_data['out_idxes'][i:]
                    tmp_batch_ids = batch_data['batch_ids'][i:]
                    for s in range(len(tmp_in_data[0])):
                        batch_in = []
                        batch_out = []
                        batch_last = []
                        batch_seq_l = []
                        for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                            _in = tmp_in[s]
                            _out = tmp_out[s] - 1
                            batch_last.append(_in)
                            batch_in.append(tmp_in[:s + 1])
                            batch_out.append(_out)
                            batch_seq_l.append(s + 1)
                        feed_dict = {
                            self.inputs: batch_in,
                            self.last_inputs: batch_last,
                            self.lab_input: batch_out,
                            self.sequence_length: batch_seq_l

                        }

                        # train
                        preds, loss, alpha = sess.run(
                            [self.softmax_input, self.loss, self.alph],
                            feed_dict=feed_dict
                        )
                        t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
                        test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                        test_data.pack_preds(ranks, tmp_batch_ids)
                        c_loss += list(loss)
                        recall += t_r
                        mrr += t_m
                        batch += 1
            else:
                tmp_in_data = batch_data['in_idxes']
                tmp_out_data = batch_data['out_idxes']
                tmp_batch_ids = batch_data['batch_ids']
                for s in range(len(tmp_in_data[0])):
                    batch_in = []
                    batch_out = []
                    batch_last = []
                    batch_seq_l = []
                    for tmp_in, tmp_out in zip(tmp_in_data, tmp_out_data):
                        _in = tmp_in[s]
                        _out = tmp_out[s] - 1
                        batch_last.append(_in)
                        batch_in.append(tmp_in[:s + 1])
                        batch_out.append(_out)
                        batch_seq_l.append(s + 1)
                    feed_dict = {
                        self.inputs: batch_in,
                        self.last_inputs: batch_last,
                        self.lab_input: batch_out,
                        self.sequence_length: batch_seq_l

                    }

                    # train
                    preds, loss, alpha = sess.run(
                        [self.softmax_input, self.loss, self.alph],
                        feed_dict=feed_dict
                    )
                    t_r, t_m, ranks = cau_recall_mrr_org(preds, batch_out, cutoff=self.cut_off)
                    test_data.pack_ext_matrix('alpha', alpha, tmp_batch_ids)
                    test_data.pack_preds(ranks, tmp_batch_ids)
                    c_loss += list(loss)
                    recall += t_r
                    mrr += t_m
                    batch += 1
        r, m =cau_samples_recall_mrr(test_data.samples,self.cut_off)
        print (r,m)
        print (np.mean(c_loss))
        return  np.mean(recall), np.mean(mrr)
