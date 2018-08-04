import pandas as pd
import numpy as np
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack


def load_data(train_file, test_file, pad_idx=0, class_num = 3):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    # the global param.
    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    idx_cnt = 0
    # load the data
    train_data, idx_cnt = _load_data(train_file, items2idx, idx_cnt, pad_idx, class_num)
    print(len(items2idx.keys()))
    test_data, idx_cnt = _load_data(test_file, items2idx, idx_cnt, pad_idx, class_num)
    print(len(items2idx.keys()))
    return train_data, test_data, items2idx



def _load_data(file_path, item2idx, idx_cnt, pad_idx, class_num):

    data = pd.read_csv(file_path, sep='\t', dtype={'ItemId': np.int64})
    print("read finish")
    # return
    data.sort_values(['SessionId', 'Time'], inplace=True)  # 按照sessionid和时间升序排列
    print("sort finish")
    # y = list(data.groupby('SessionId'))
    print("list finish")
    # tmp_data = dict(y)

    samplepack = Samplepack()
    samples = []
    now_id = 0
    print("I am reading")
    sample = Sample()
    last_id = None
    click_items = []
    for s_id,item_id in zip(list(data['SessionId'].values),list(data['ItemId'].values)):
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            item_dixes = []
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])
            in_dixes = item_dixes[:-1]
            out_dixes = item_dixes[1:]
            sample.id = now_id
            sample.session_id = last_id
            sample.click_items = click_items
            sample.items_idxes = item_dixes
            sample.in_idxes = in_dixes
            sample.out_idxes = out_dixes
            samples.append(sample)
            # print(sample)
            sample = Sample()
            last_id =s_id
            click_items = []
            now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
        # click_items = list(tmp_data[session_tmp_idx]['ItemId'])
    sample = Sample()
    item_dixes = []
    for item in click_items:
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])
    in_dixes = item_dixes[:-1]
    out_dixes = item_dixes[1:]
    sample.id = now_id
    sample.session_id = last_id
    sample.click_items = click_items
    sample.items_idxes = item_dixes
    sample.in_idxes = in_dixes
    sample.out_idxes = out_dixes
    samples.append(sample)
    print(sample)


    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt





# root_path = '/home/herb/code'
# project_name = '/Basic'
# res_train_path = root_path + project_name +  '/datas/data/Restaurants_Train.xml'
# res_test_path = root_path + project_name +  '/datas/data/Restaurants_Test_Gold.xml'
# lap_train_path = root_path + project_name +  '/datas/data/Laptops_Train.xml'
# lap_test_path = root_path + project_name +  '/datas/data/Laptops_Test_Gold.xml'
#
# if __name__ == '__main__':
#     train_data, test_data, word2idx = load_data(
#         lap_train_path,
#         lap_test_path,
#         class_num=2
#     )
#     print len(train_data.samples)
#     print len(test_data.samples)

#
# mid_res_train_data = "res_train.data"
# mid_res_test_data = "res_test.data"
# mid_res_emb_dict = "res_emb_dict.data"
# mid_res_word2idx = "res_mid_word2idx.data"
# mid_lap_train_data = "lap_train.data"
# mid_lap_test_data = "lap_test.data"
# mid_lap_emb_dict = "lap_emb_dict.data"
# mid_lap_word2idx = "lap_mid_word2idx.data"
# mid_res_cat_emb_dict = "datas/res_cat_emb_dict.data"
# mid_res_cat_word2idx = "datas/res_cat_mid_word2idx.data"
# if __name__ == '__main__':
#     trian, test, word2idx = load_data("../datas/data/Laptops_Train.xml",
#                                       "../datas/data/Laptops_Test_Gold.xml")
#     dataset = []
#
#     tr_d = trian.samples
#     te_d = test.samples
#         #  print tr_d
#     data = tr_d + te_d
#         #  print data
#     dataset += data
#     for i in range(len(dataset)):
#         dataset[i].id = i
#     for x in dataset:
#         print x.id
#     rand_idx = np.random.permutation(len(dataset))
#     alenth = len(rand_idx) / 5
#     count = 0
#     datas = []
#     for i in xrange(5):
#         cdatas = []
#         for j in xrange(alenth):
#             cdatas.append(dataset[rand_idx[count]])
#             count += 1
#         datas.append(cdatas)
#         # solve last data
#     for i in rand_idx[count:]:
#             datas[-1].append(dataset[i])
#
#     tmp = copy.deepcopy(datas)
#     for i in xrange(5):
#         datas = copy.deepcopy(tmp)
#         test_datas = datas.pop(i)
#         trian_datas = []
#         path = "../datas/cross" + str(i + 1) + "/"
#         for x in datas:
#             trian_datas += x
#         print len(test_datas)
#         print len(trian_datas)
#         samplepack_train = Samplepack()
#         samplepack_train.samples = trian_datas
#         samplepack_train.init_id2sample()
#         samplepack_test = Samplepack()
#         samplepack_test.samples = test_datas
#         samplepack_test.init_id2sample()
#         dump_file(
#             [samplepack_train, path + mid_lap_train_data],
#             [samplepack_test, path + mid_lap_test_data]
#         )



PATH_TO_TRAIN = 'G:/PyCharm 2016.2.3/GRU4Rec-master/GRU4Rec-master/processed/data/rsc15_train_full.txt'
PATH_TO_TEST = 'G:/PyCharm 2016.2.3/GRU4Rec-master/GRU4Rec-master/processed/data/rsc15_test.txt'

if __name__ =='__main__':
    _load_data(PATH_TO_TRAIN,{},0,0,0)