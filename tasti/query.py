import tasti
import sklearn
import numpy as np
import pandas as pd
import supg.datasource as datasource
from tqdm.autonotebook import tqdm
from blazeit.aggregation.samplers import ControlCovariateSampler, TrueSampler
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector, ImportancePrecisionTwoStageSelector
from tabulate import tabulate
from numba import njit, jit, prange

def print_dict(d, header='Key'):
    headers = [header, '']
    data = [(k,v) for k,v in d.items()]
    print(tabulate(data, headers=headers))

@jit(parallel=True)
def propagation(y_pred, topk_reps, topk_distances):
    '''
    numba acceleration
    topk_reps here is a sub-array of the y_true
    '''
    for i in prange(len(y_pred)):
        weights = topk_distances[i]
        weights = np.sum(weights) - weights
        weights = weights / weights.sum()
        counts = topk_reps[i]
        y_pred[i] =  np.sum(counts * weights)
    return y_pred

class BaseQuery:
    def __init__(self, index):
        self.index = index
        self.df = False

    def score(self, target_dnn_output):
        raise NotImplementedError

    def propagate(self, target_dnn_cache, reps, topk_reps, topk_distances):
        if not self.df:
            score_fn = self.score
            y_true = np.array(
                [tasti.DNNOutputCacheFloat(target_dnn_cache, score_fn, idx) for idx in range(len(topk_reps))]
            )
            y_pred = np.zeros(len(topk_reps))

            '''for i in tqdm(range(len(y_pred)), 'Propagation'):
                weights = topk_distances[i]
                weights = np.sum(weights) - weights
                weights = weights / weights.sum()
                counts = y_true[topk_reps[i]]
                y_pred[i] =  np.sum(counts * weights)'''
            for i in reps:
                y_true[i] = float(y_true[i])
            tmp = y_true[topk_reps]
            tmp = tmp.astype(np.float32)
            y_pred = propagation(y_pred, tmp, topk_distances)
        else:
            y_true = self.score(target_dnn_cache.df)
            y_pred = np.zeros(len(topk_reps))  
            weights = topk_distances
            weights = np.sum(weights, axis=1).reshape(-1, 1) - weights
            weights = weights / weights.sum(axis=1).reshape(-1, 1)
            counts = np.take(y_true, topk_reps)
            y_pred = np.sum(counts * weights, axis=1)
        return y_pred, y_true

    def execute(self):
        raise NotImplementedError

class AggregateQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, err_tol=0.01, confidence=0.05, y=None):
        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        r = max(1, np.amax(np.rint(y_pred)))
        print("r", r)
        sampler = ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r)
        estimate, nb_samples = sampler.sample()

        res = {
            'initial_estimate': y_pred.sum(),
            'debiased_estimate': estimate,
            'nb_samples': nb_samples,
            'y_pred': y_pred,
            'y_true': y_true
        }
        return res

    def execute(self, err_tol=0.01, confidence=0.05, y=None):
        res = self._execute(err_tol, confidence, y)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, err_tol=0.01, confidence=0.05, y=None):
        res = self._execute(err_tol, confidence, y)
        res['actual_estimate'] = res['y_true'].sum() # expensive
        print_dict(res, header=self.__class__.__name__)
        return res

class LimitQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)

    def execute(self, want_to_find=5, nb_to_find=10, GAP=300, y=None):
        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        for ind in order:
            if ind in visited:
                continue
            nb_calls += 1
            if float(y_true[ind]) >= want_to_find:
                ret_inds.append(ind)
                for offset in range(-GAP, GAP+1):
                    visited.add(offset + ind)
            if len(ret_inds) >= nb_to_find:
                break
        res = {
            'nb_calls': nb_calls,
            'ret_inds': ret_inds
        }
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, want_to_find=5, nb_to_find=10, GAP=300, y=None):
        return self.execute(want_to_find, nb_to_find, GAP, y)

class SUPGPrecisionQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _execute(self, budget, y=None):
        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=0.95, delta=0.05,
            budget=budget
        )
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
        inds = selector.select()

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source
        }

        return res

    def execute(self, budget, y=None):
        res = self._execute(budget, y)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, y=None):
        res = self._execute(budget, y)
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res

class SUPGRecallQuery(SUPGPrecisionQuery):
    def _execute(self, budget, y=None):
        if y == None:
            y_pred, y_true = self.propagate(
                self.index.target_dnn_cache,
                self.index.reps, self.index.topk_reps, self.index.topk_dists
            )
        else:
            y_pred, y_true = y

        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='rt',
            min_recall=0.90, min_precision=0.90, delta=0.05,
            budget=budget
        )
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt')
        inds = selector.select()

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source
        }
        return res

    def execute(self, budget, y=None):
        res = self._execute(budget, y)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, y=None):
        res = self._execute(budget, y)
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res
