import pandas as pd
import numpy as np
import json
import csv
import os

# 评分预测    1-5
class MatrixDecomForRecSys_filter(object):
    def __init__(
            self, 
            lr,
            batch_size,
            reg_p, 
            reg_q, 
            hidden_size, 
            epoch, 
            columns=["uid", "iid", "rating"],
            metric=None,
            ):
        self.lr = lr # 学习率
        self.batch_size = batch_size
        self.reg_p = reg_p    # P矩阵正则系数
        self.reg_q = reg_q    # Q矩阵正则系数
        self.gamma = 0.01
        self.hidden_size = hidden_size  # 隐向量维度
        self.epoch = epoch    # 最大迭代次数
        self.columns = columns
        self.metric = metric

    
    def load_dataset(self, train_data, dev_data):
        self.train_data = pd.DataFrame(train_data)
        self.dev_data = pd.DataFrame(dev_data)

        self.users_ratings = train_data.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = train_data.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.train_data[self.columns[2]].mean()
    

    def _init_matrix(self):
        '''
        *********************************
        用户矩阵P和物品矩阵Q的初始化也对算法优化有一定帮助，更好的初始化相当于先验信息。
        加分项：
        - 思考初始化的一些方法，正态分布等等；
        - 其他初始化方法？
        *********************************
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.hidden_size).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.hidden_size).astype(np.float32)
        ))

        return P, Q

    def Pearson(self, u1, u2):
        e, sigma, sigma1, sigma2 = 0, 0, 0, 0
        row1, row2 = self.users_ratings.loc[u1], self.users_ratings.loc[u2]
        mean1, mean2 = np.mean(row1['movieId'].values[0]), np.mean(row2['movieId'].values[0])
        for i, v1 in enumerate(row1['movieId'].values[0]):
            for j, v2 in enumerate(row2['movieId'].values[0]):
                if(v1 == v2):
                    e += (row1['rating'].values[0][i] - mean1) * (row2['rating'].values[0][j] - mean2)
                    sigma1 += (row1['rating'].values[0][i] - mean1) * (row1['rating'].values[0][i] - mean1)
                    sigma2 += (row2['rating'].values[0][j] - mean2) * (row2['rating'].values[0][j] - mean2)
        sigma = np.sqrt(sigma1) * np.sqrt(sigma2)
        return e / sigma

    def train(self, optimizer_type: str):
        '''
        训练模型
        :param dataset: uid, iid, rating
        :return:
        '''
        P, Q = self._init_matrix() # 初始化user、item矩阵
        best_metric_result = None
        best_P, best_Q = P, Q
        self.sim = {}
        for u1, _ in self.users_ratings.iterrows():
                for u2, _ in self.users_ratings.iterrows():
                    if u1 < u2:
                        self.sim[(u1, u2)] = self.Pearson(u1, u2)

        for i in range(self.epoch):
            print("Epoch: %d"%i)
            # 当前epoch，执行优化算法：
            if optimizer_type == "SGD": # 随机梯度下降
                P, Q = self.sgd(P, Q)
            elif optimizer_type == "BGD": # 批量梯度下降
                P, Q = self.bgd(P ,Q, batch_size=self.batch_size)
            else:
                raise NotImplementedError("Please choose one of SGD and BGD.")
            # 当前epoch优化后，在验证集上验证，并保存目前最好的P和Q
            metric_result = self.eval(P, Q)
            # 如果当前的RMSE更低，则保存
            print("Current dev metric result: {}".format(metric_result))
            if best_metric_result is None or metric_result <= best_metric_result:
                best_metric_result = metric_result
                best_P, best_Q = P, Q
                print("Best dev metric result: {}".format(best_metric_result))

        # 最后保存最好的P和Q
        np.savez("best_pq.npz", P=best_P, Q=best_Q)


    def sgd(self, P, Q):
        '''
        *********************************
        基本分：请实现【随机梯度下降】优化
        加分项：进一步优化如下
        - 考虑偏置项
        - 考虑正则化
        - 考虑协同过滤
        *********************************
        '''
        sum_ = np.zeros(self.hidden_size)
        for key, value in self.sim.items():
            sum_ += value * (P[key[0]] - P[key[1]])

        for uid, iid, real_rating in self.train_data.itertuples(index=False):
            p_u, q_i = P[uid], Q[iid]
            err = np.float32(real_rating - np.dot(p_u, q_i))
            p_u += self.lr * (err * q_i - self.reg_p * p_u - self.gamma * sum_)
            q_i += self.lr * (err * p_u - self.reg_q * q_i)
            P[uid] = p_u
            Q[iid] = q_i
        return P, Q

    def bgd(self, P, Q, batch_size: int=8):
        '''
        *********************************
        基本分：请实现【批量梯度下降】优化
        加分项：进一步优化如下
        - 考虑偏置项
        - 考虑正则化
        - 考虑协同过滤
        *********************************
        '''
        #训练集划分成若干互不重叠且大小为batch_size的批量数据集
        sum_ = np.zeros(self.hidden_size)
        for key, value in self.sim.items():
            sum_ += value * (P[key[0]] - P[key[1]])

        num_batches = len(self.train_data) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            data = self.train_data[batch_start : batch_end]
            for uid, iid, real_rating in data.itertuples(index=False):
                p_u, q_i = P[uid], Q[iid]
                err = np.float32(real_rating - np.dot(p_u, q_i))
                p_u += self.lr * (err * q_i - self.reg_p * p_u - self.gamma * sum_)
                q_i += self.lr * (err * p_u - self.reg_q * q_i)
                P[uid] = p_u
                Q[iid] = q_i
        return P, Q

    def predict_user_item_rating(self, uid, iid, P, Q):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = P[uid]
        q_i = Q[iid]

        return np.dot(p_u,q_i)
    
    def eval(self, P, Q):
        # 根据当前的P和Q，在dev上进行验证，挑选最好的P和Q向量
        dev_loss = 0.
        prediction, ground_truth = list(), list()
        for uid, iid, real_rating in self.dev_data.itertuples(index=False):
            prediction_rating = self.predict_user_item_rating(uid, iid, P, Q)
            # dev_loss += abs(prediction_rating - real_rating)
            prediction.append(prediction_rating)
            ground_truth.append(real_rating)
        
        metric_result = self.metric(ground_truth, prediction)
        
        return metric_result


    def test(self, test_data):
        '''预测测试集榜单数据'''
        # 预测结果可以提交至：https://www.kaggle.com/competitions/dase-recsys/overview
        test_data = pd.DataFrame(test_data)
        # 加载训练好的P和Q
        best_pq = np.load("best_pq.npz", allow_pickle=True)
        P, Q = best_pq["P"][()], best_pq["Q"][()]

        save_results = list()
        for _, uid, iid in test_data.itertuples(index=False):
            pred_rating = self.predict_user_item_rating(uid, iid, P, Q)
            save_results.append(pred_rating)
        
        log_path = "submit_results.csv"
        if os.path.exists(log_path):
            os.remove(log_path)
        file = open(log_path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'ID', 'rating'])
        for ei, rating in enumerate(save_results):
            csv_writer.writerow([ei, rating])
        file.close()
