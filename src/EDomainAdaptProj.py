import os
import copy
import sys

from termcolor import colored
import numpy as np
import statistics

from CEPC.src.EDomainAdaptMine1 import EDomainAdaptMine1
from CEPC.src.EDomainAdaptSharedProj import EDACMD
from CEPC.src.ELib import ELib
from CEPC.src.ELblConf import ELblConf
from CEPC.src.ELbl import ELbl
from CEPC.src.EVar import EVar
from CEPC.src.EBertUtils import EInputBundle, EDomainAdaptParam


class EDomainAdaptProj:

    @staticmethod
    def __print_iteration_results(queries, itr_average=None, itr_detail=None):
        if itr_average is None:
            itr_average = copy.deepcopy(itr_detail[0])
            for entry in itr_detail[1:]:
                itr_average += entry
            itr_average /= len(itr_detail)
        if itr_detail is None or itr_detail.shape[0] == 1:
            for ind, cur_q in enumerate(queries):
                cur_row = itr_average[ind]
                print('{: <25} L1> F1: {:.3f} Pre: {:.3f} Rec: {:.3f} Acc: {:.3f}'.format(
                    cur_q, cur_row[0], cur_row[1], cur_row[2], cur_row[3]))
            ave = itr_average.mean(axis=0)
            print('_____')
            print('{: <25} L1> F1: {:.3f} Pre: {:.3f} Rec: {:.3f} Acc: {:.3f}'.format(
                'average', ave[0], ave[1], ave[2], ave[3]))
        else:
            for ind, cur_q in enumerate(queries):
                cur_row = itr_average[ind]
                print('{: <25} L1> '
                      'F1: {:.3f}+/-{:.3f}   Pre: {:.3f}+/-{:.3f}   Rec: {:.3f}+/-{:.3f}   Acc: {:.3f}+/-{:.3f}'.format(
                    cur_q,
                    cur_row[0], statistics.stdev(itr_detail[:, ind, 0]),
                    cur_row[1], statistics.stdev(itr_detail[:, ind, 1]),
                    cur_row[2], statistics.stdev(itr_detail[:, ind, 2]),
                    cur_row[3], statistics.stdev(itr_detail[:, ind, 3])))
            ave = itr_average.mean(axis=0)
            print('_____')
            print('{: <25} L1> '
                  'F1: {:.3f}+/-{:.3f}   Pre: {:.3f}+/-{:.3f}   Rec: {:.3f}+/-{:.3f}   Acc: {:.3f}+/-{:.3f}'.format(
                'average',
                ave[0], statistics.stdev(itr_detail.mean(axis=1)[:, 0]),
                ave[1], statistics.stdev(itr_detail.mean(axis=1)[:, 1]),
                ave[2], statistics.stdev(itr_detail.mean(axis=1)[:, 2]),
                ave[3], statistics.stdev(itr_detail.mean(axis=1)[:, 3])))
        ELib.PASS()

    @staticmethod
    def __run_multi_source(cmd, lc, cur_itr, model_path, output_dir, device, device_2, seed,
                           src_labeled, src_unlabeled, tgt_labeled, tgt_unlabeled, param):
        result = None
        if cmd == EDACMD.da_m_mine1:
            result = EDomainAdaptProj.__da_multi_mine1(cur_itr, lc, model_path, output_dir, device, device_2,
                seed, src_labeled, src_unlabeled, tgt_labeled, tgt_unlabeled, param)
        return result

    @staticmethod
    def __run_one_iteration(cmd, lc, cur_itr, model_path, output_dir, device, device_2, seed,
                            labeled_bundles, unlabeled_bundles, tgt_d, src_d, param):
        topics = list()
        results = list()
        for cur_ind, _ in enumerate(labeled_bundles):
            src_labeled = labeled_bundles[:cur_ind] + labeled_bundles[cur_ind + 1:]
            src_unlabeled = unlabeled_bundles[:cur_ind] + unlabeled_bundles[cur_ind + 1:]
            tgt_labeled = labeled_bundles[cur_ind]
            tgt_unlabeled = unlabeled_bundles[cur_ind]
            if tgt_d is not None and tgt_labeled.tws[0].Query.lower() != tgt_d.lower():
                continue
            if EDACMD.is_multi(cmd):
                print(colored('processing ' + tgt_labeled.tws[0].Query, 'red'))
                topics.append(tgt_labeled.tws[0].Query)
                cur_result = EDomainAdaptProj.__run_multi_source(cmd, lc, cur_itr, model_path, output_dir, device,
                                                                 device_2, seed, src_labeled, src_unlabeled,
                                                                 tgt_labeled, tgt_unlabeled, param)
                results.append(cur_result)
            else:
                print(colored('Unknown command.\n', 'red'))
                sys.exit(0)
            ELib.PASS()
        if None in results:
            print(colored('The pre-computation was done. The cache files were stored. Re-run the program.\n', 'red'))
            sys.exit(0)
        results = np.array(results)
        print()
        EDomainAdaptProj.__print_iteration_results(topics, itr_average=results)
        print()
        return topics, results

    @staticmethod
    def __da_multi_mine1(cur_itr, lc, model_path, output_dir, device, device_2, seed,
                         src_labeled, src_unlabeled, tgt_labeled, tgt_unlabeled, param):
        result = EDomainAdaptMine1.run(cur_itr, lc, model_path, output_dir, device, device_2, seed,
                                       src_labeled, src_unlabeled, tgt_labeled, tgt_unlabeled, param)
        return result

    @staticmethod
    def run(cmd, itr, model_path, data_path, output_dir, device, device_2, seed, tgt_d, src_d, cache_dir, flag):
        EVar.BertBatchSize = 50
        EVar.MaxSequence = None # for short docs: 160 # for long docs: 160 * 2 # for automatic: None
        is_tuning = False
        tuning_rep = 2

        lc = ELblConf(0, 1, [ELbl(0, EVar.LblNonEventHealth), ELbl(1, EVar.LblEventHealth)])
        if is_tuning:
            itr *= tuning_rep
        labeled_bundles = EInputBundle.get_tweet_query_bundles(2, lc, data_path)
        unlabeled_bundles = EInputBundle.get_tweet_query_bundles(2, lc, data_path, remove_lbls=True)

        if EVar.MaxSequence is None:
            EVar.MaxSequence = 160
            for cur_bundle in labeled_bundles:
                d_lens, d_max, d_min, d_ave = ELib.get_doc_length_stat(cur_bundle, model_path)
                if d_ave >= 160:
                    EVar.MaxSequence = 160 * 2
                    break
            ELib.PASS()
        param = EDomainAdaptParam()
        param.cache_dir = cache_dir
        param.cache_global = dict()
        param.flag = flag
        result = None
        initial_seed = seed
        print('setting:', '| batch-size>', EVar.BertBatchSize, '| max-seq-len>', EVar.MaxSequence)
        for cur_itr in range(itr):
            print(colored('------------------------------------', 'red'))
            print('iteration {}/{} began with seed=\'{}\'   at {}'.format(cur_itr + 1, itr, seed, ELib.get_time()))
            param.cache_iteration = dict()
            cur_output_dir = output_dir + '_' + str(cur_itr) + '/'
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            qs, cur_results = EDomainAdaptProj.__run_one_iteration(cmd, lc, cur_itr, model_path, cur_output_dir, device,
                device_2, seed, labeled_bundles, unlabeled_bundles, tgt_d, src_d, param)
            cur_results = np.expand_dims(cur_results, 0)
            result = cur_results if result is None else np.append(result, cur_results, 0)
            if is_tuning:
                if (cur_itr + 1) % tuning_rep == 0:
                    print(colored('====================================', 'red'))
                    EDomainAdaptProj.__print_iteration_results(qs, itr_detail=result)
                    print(param.report())
                    print(colored('====================================', 'red'))
                    result = None
                    param.next_param()
                    seed = initial_seed
                else:
                    seed += 1230
            else:
                seed += 1230
        if not is_tuning:
            print(colored('====================================', 'red'))
            print('Final Results:')
            EDomainAdaptProj.__print_iteration_results(qs, itr_detail=result)
        ELib.PASS()



