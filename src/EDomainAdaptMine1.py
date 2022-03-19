import gc
import copy
import math
import statistics
import sys

from sklearn.linear_model import LogisticRegression
from termcolor import colored
import numpy as np
from scipy.spatial import distance
from sklearn import metrics

import torch
import torch.nn.functional as F

from CEPC.src.ELib import ELib
from CEPC.src.ETweet import ETweet
from CEPC.src.EBert import EBert, EBertCLSType
from CEPC.src.EBertUtils import EBertConfig, EBalanceBatchMode, EInputListMode, EInputBundle


class EDomainAdaptMine1:

    @staticmethod
    def __print_teacher_labels(src_labeled, lbl_pred, prob_pred, topic, output_dir, file_name=None):
        if file_name is None:
            file_name = topic
        prob_pred = np.array(prob_pred)
        d_probs, d_topics = list(), list()
        print('"{}" clusters> '.format(topic), end='')
        for src_ind in range(len(src_labeled)):
            print('"{}": {} | '.format(src_labeled[src_ind].tws[0].Query, lbl_pred.count(src_ind)), end='')
            d_probs.append(prob_pred[:, src_ind])
            d_topics.append(src_labeled[src_ind].tws[0].Query)
        print()

    @staticmethod
    def __domain_test(cls, src_labeled, tgt_labeled):
        topic = tgt_labeled.tws[0].Query
        domain_count = len(src_labeled)
        for d_ind in range(0, domain_count):
            cls.bert_classifier.active_domain = d_ind
            cls.test(tgt_labeled, title=src_labeled[d_ind].tws[0].Query + '>' + topic, report_number_of_intervals=10)
        ELib.PASS()

    @staticmethod
    def __get_cls(model_path, output_dir, device, device_2, seed, src_labeled, tgt_unlabeled):
        config = EBertConfig.get_config(None, EBertCLSType.coordinated, model_path, None,
                                        None, None, None, output_dir, 3, device, device_2, seed, None,
                                        check_early_stopping=False)
        cls = EBert(config, domain_count=len(src_labeled))
        cls.custom_train_loss_func = EDomainAdaptMine1.__train_loss
        cls.custom_test_loss_func = EDomainAdaptMine1.__test_loss
        return cls

    @staticmethod
    def __replicate_short_sets(bundles, lc, seed, extra_max=None):
        max_len = -1
        for cur_b in bundles:
            if len(cur_b.tws) > max_len:
                max_len = len(cur_b.tws)
        if extra_max is not None:
            max_len = max(max_len, extra_max)
        for cur_b in bundles:
            if len(cur_b.tws) < max_len:
                EInputBundle.populate_bundle(cur_b, max_len, seed, lc)
        ELib.PASS()

    @staticmethod
    def __scores_coral(vectors, src_topics, device):
        ## https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
        ## https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
        result = dict()
        for d_ind, cur_dom_vec in enumerate(vectors):
            src = torch.tensor(cur_dom_vec['src']).to(device)
            tgt = torch.tensor(cur_dom_vec['tgt']).to(device)
            ## source covariance mat
            src_scores_sum = src.size(0)
            src_scores_normal = torch.tensor([1.0 / src_scores_sum for _ in range(src.size(0))]).to(device).view(-1, 1)
            src_scores_normal_squared_sum = torch.sum(src_scores_normal * src_scores_normal).item()
            src_weighted_sum = src_scores_normal * src
            src_mean = torch.sum(src_weighted_sum, dim=0).view(1, -1)
            src_cov = 0
            for src_ind in range(src.size(0)):
                diff_to_mean = src[src_ind, :].view(1, -1) - src_mean
                diff_to_mean_t = diff_to_mean.t()
                cur_cov = src_scores_normal[src_ind] * diff_to_mean_t.matmul(diff_to_mean)
                src_cov += cur_cov
            src_cov = src_cov / (1 - src_scores_normal_squared_sum)
            ## target covariance mat
            tgt_scores_sum = tgt.size(0)
            tgt_scores_normal = torch.tensor([1.0 / tgt_scores_sum for _ in range(tgt.size(0))]).to(device).view(-1, 1)
            tgt_scores_normal_squared_sum = torch.sum(tgt_scores_normal * tgt_scores_normal).item()
            tgt_weighted_sum = tgt_scores_normal * tgt
            tgt_mean = torch.sum(tgt_weighted_sum, dim=0).view(1, -1)
            cur_result = list()
            for tgt_ind in range(tgt.size(0)):
                diff_to_mean = tgt[tgt_ind, :].view(1, -1) - tgt_mean
                diff_to_mean_t = diff_to_mean.t()
                cur_cov = tgt_scores_normal[tgt_ind] * diff_to_mean_t.matmul(diff_to_mean)
                cur_cov = cur_cov / (1 - tgt_scores_normal_squared_sum) # normalize
                diff = src_cov - cur_cov
                doc_coral_loss = torch.norm(diff * diff, p='fro')
                cur_result.append(doc_coral_loss.item())
            result[src_topics[d_ind]] = cur_result
        return result

    @staticmethod
    def __scores_density(vectors, src_topics, seed):
        result = dict()
        for d_ind, cur_dom_vec in enumerate(vectors):
            src = cur_dom_vec['src']
            tgt = cur_dom_vec['tgt']
            ## log reg
            train_vecs = list()
            train_lbls = list()
            train_vecs.extend(src)
            train_lbls.extend([0 for _ in src])
            train_vecs.extend(tgt)
            train_lbls.extend([1 for _ in tgt])
            lr = LogisticRegression(random_state=seed).fit(train_vecs, train_lbls)
            probs = lr.predict_proba(tgt)
            cur_result = list()
            for cur_prob in probs:
                cur_result.append((len(tgt) / len(src)) * (cur_prob[0] / cur_prob[1]))
            result[src_topics[d_ind]] = cur_result
        return result

    @staticmethod
    def __scores_labels(cls, tgt_unlabeled, src_topics):
        result_lbl = dict()
        result_probs = dict()
        for d_ind, cur_topic in enumerate(src_topics):
            cls.bert_classifier.active_domain = d_ind
            cur_lbls, cur_logits, _, _ = cls.test(tgt_unlabeled, title=cur_topic + '>' + tgt_unlabeled.tws[0].Query,
                                                  report_number_of_intervals=10, print_perf=False)
            result_lbl[cur_topic] = cur_lbls
            result_probs[cur_topic] = F.softmax(torch.tensor(cur_logits), dim=1).numpy().tolist()
        return result_lbl, result_probs

    @staticmethod
    def __scores(cls, data, src_labeled, tgt_unlabeled, coral_scales):
        result = dict()
        cls.bert_classifier.backup_domain_modules()
        topics = [src_d.tws[0].Query for src_d in src_labeled]
        ## train the src classifiers with cross entropy
        print(colored('training the src classifiers using cross entropy...', 'green'))
        cls.bert_classifier.train_state = 2
        cls.config.epoch_count = 3
        cls.bert_classifier.train_step = -1
        batch_modes = [EBalanceBatchMode.label_based for _ in range(len(src_labeled))] + [EBalanceBatchMode.none]
        cls.train(data, input_mode=EInputListMode.parallel, balance_batch_mode_list=batch_modes)
        ## collect the vectors and calculate the coral scores
        print(colored('collecting the vectors to calculate coral scores...', 'green'))
        cls.bert_classifier.train_state = 1
        cls.config.epoch_count = 1
        cls.bert_classifier.train_step = -1
        cls.train(data, input_mode=EInputListMode.parallel_full, switch_on_train_mode=False, train_shuffle=False,
                  train_drop_last=False)
        result['coral_scores'] = EDomainAdaptMine1.__scores_coral(cls.bert_classifier.vectors, topics, cls.config.device)
        ## train the src classifiers with different coral scales and store the info
        for cur_scale in coral_scales:
            ## reset the settings
            cls.bert_classifier.restore_domain_modules()
            cls.bert_classifier.clear_vectors()
            ## train the src classifiers with coral loss
            print(colored(ELib.get_time() + ' training the src classifiers using coral loss with scale:{} ...'.
                          format(cur_scale), 'green'))
            result[cur_scale] = dict()
            cls.bert_classifier.coral_scale = [cur_scale for _ in topics]
            cls.bert_classifier.train_state = 3
            cls.config.epoch_count = 3
            cls.bert_classifier.train_step = -1
            batch_modes = [EBalanceBatchMode.label_based for _ in range(len(src_labeled))] + [EBalanceBatchMode.none]
            cls.train(data, input_mode=EInputListMode.parallel, balance_batch_mode_list=batch_modes)
            ## collect the vectors and calculate the density scores
            print(colored('collecting the vectors to calculate density scores...', 'green'))
            cls.bert_classifier.train_state = 1
            cls.config.epoch_count = 1
            cls.bert_classifier.train_step = -1
            cls.train(data, input_mode=EInputListMode.parallel_full, switch_on_train_mode=False, train_shuffle=False,
                      train_drop_last=False)
            result[cur_scale]['density_scores'] = EDomainAdaptMine1.__scores_density(
                cls.bert_classifier.vectors, topics, cls.config.seed)
            ## label the tgt data and store it
            print(colored('labeling the tgt data using the src classifiers...', 'green'))
            result[cur_scale]['labels'], result[cur_scale]['probs'] = \
                EDomainAdaptMine1.__scores_labels(cls, tgt_unlabeled, topics)
        ## reset the settings
        cls.bert_classifier.restore_domain_modules()
        cls.bert_classifier.clear_vectors()
        return result

    @staticmethod
    def __get_scales_and_encoders_target_preds(src_labeled, scores, coral_scales, context, js_indices):
        result = 0
        for pred_ind in range(len(src_labeled)):
            ## access the prediction domain labels
            pred_topic = src_labeled[pred_ind].tws[0].Query
            pred_cur = js_indices[pred_ind]
            pred_js_ind = context[pred_ind]['js-sorted'][pred_cur]
            pred_coral = coral_scales[pred_js_ind]
            pred_lbls = scores[pred_coral]['labels'][pred_topic] # scores[str(pred_coral)]['labels'][pred_topic]
            sum_pred_f1 = 0
            for gold_ind in range(len(src_labeled)):
                if pred_ind != gold_ind:
                    ## access the gold domain labels
                    gold_topic = src_labeled[gold_ind].tws[0].Query
                    gold_cur = js_indices[gold_ind]
                    gold_js_ind = context[gold_ind]['js-sorted'][gold_cur]
                    gold_coral = coral_scales[gold_js_ind]
                    gold_lbls = scores[gold_coral]['labels'][gold_topic] # scores[str(gold_coral)]['labels'][gold_topic]
                    cur_pred_f1 = metrics.f1_score(gold_lbls, pred_lbls)
                    sum_pred_f1 += cur_pred_f1
            result += (sum_pred_f1 / (len(src_labeled) - 1))
        return result

    @staticmethod
    def __get_scales_and_encoders(lc, src_labeled, tgt_unlabeled, scores, coral_scales):
        context = list()
        ## create a data struct and store the src label distros and JS distances for each (domain, coral_scale)
        for d_ind, cur_dom in enumerate(src_labeled):
            context.append(dict())
            src_neg_count = len(ETweet.filter_tweets_by_correct_label(cur_dom.tws, lc, lc.negative_new_label.new_label))
            src_pos_count = len(ETweet.filter_tweets_by_correct_label(cur_dom.tws, lc, lc.positive_new_label.new_label))
            context[-1]['src-dist'] = [src_neg_count/len(cur_dom.tws), src_pos_count/len(cur_dom.tws)]
            context[-1]['tgt-dist'] = list()
            cur_topic = cur_dom.tws[0].Query
            for cur_scale in coral_scales:
                cur_lbls = scores[cur_scale]['labels'][cur_topic] # scores[str(cur_scale)]['labels'][cur_topic]
                cur_neg_count = cur_lbls.count(lc.negative_new_label.new_label)
                cur_pos_count = cur_lbls.count(lc.positive_new_label.new_label)
                cur_distro = [cur_neg_count / len(cur_lbls), cur_pos_count / len(cur_lbls)]
                context[-1]['tgt-dist'].append(cur_distro)
        for d_ind, cur_dom in enumerate(src_labeled):
            context[d_ind]['js'] = list()
            for scale_ind, cur_scale in enumerate(coral_scales):
                context[d_ind]['js'].append(distance.jensenshannon(
                    context[d_ind]['src-dist'], context[d_ind]['tgt-dist'][scale_ind]))
            context[d_ind]['js-sorted'] = list(np.argsort(context[d_ind]['js']))
            ELib.PASS()
        ## find the coral_scales that result in max correlation between target predictions
        best_inds = [0 for dom in src_labeled]
        best_value = EDomainAdaptMine1.__get_scales_and_encoders_target_preds(
            src_labeled, scores, coral_scales, context, best_inds)
        while True:
            updates = list()
            for cur_ind in range(len(src_labeled)):
                cur_best_inds = copy.deepcopy(best_inds)
                cur_best_inds[cur_ind] += 1
                if cur_best_inds[cur_ind] >= len(coral_scales): # len(src_labeled): (BUG!)
                    continue
                cur_best_value = EDomainAdaptMine1.__get_scales_and_encoders_target_preds(
                    src_labeled, scores, coral_scales, context, cur_best_inds)
                updates.append({'value': cur_best_value, 'inds': cur_best_inds})
            if len(updates) == 0:
                break
            cur_max = max(updates, key=lambda comp: comp['value'])
            if cur_max['value'] > best_value and abs(cur_max['value'] - best_value) > 0.005:
                best_inds = cur_max['inds']
                best_value = cur_max['value']
            else:
                break
        ## map the indices to the coral scales
        result_scales = list()
        result_encoders = list()
        encoder_dict = dict()
        for cur_ind in range(len(src_labeled)):
            js_ind = context[cur_ind]['js-sorted'][best_inds[cur_ind]]
            result_scales.append(coral_scales[js_ind])
            if coral_scales[js_ind] not in encoder_dict:
                encoder_dict[coral_scales[js_ind]] = len(encoder_dict)
            result_encoders.append(encoder_dict[coral_scales[js_ind]])
        ## print
        print('coral encoders and scales> ', end='')
        for d_ind, cur_dom in enumerate(src_labeled):
            print('{}: [{}],{}   '.format(cur_dom.tws[0].Query, result_encoders[d_ind], result_scales[d_ind]), end='')
        print()
        return result_scales, result_encoders

    @staticmethod
    def __score_to_lbl(model, scores, src_labeled, tgt_unlabeled):
        ## init
        class_count = len(tgt_unlabeled.input_y_row[0][0])
        topics = [cur_src.tws[0].Query for cur_src in src_labeled]
        all_coral_scores = [scores['coral_scores'][cur_topic] for cur_topic in topics]
        max_coral_score = np.max(np.array(all_coral_scores))
        tw_count = len(scores['coral_scores'][topics[0]])
        ## construct meta_input
        input_labels = list()
        input_meta = list()
        for cur_d, cur_topic in enumerate(topics):
            cur_coral_score = model.bert_classifier.coral_scale[cur_d] # str(model.bert_classifier.coral_scale[cur_d])
            input_labels.append(scores[cur_coral_score]['labels'][cur_topic])
            coral_scores = scores['coral_scores'][cur_topic]
            density_scores = scores[cur_coral_score]['density_scores'][cur_topic]
            input_meta.append(list())
            for tw_ind in range(tw_count):
                normalized_coral_score = coral_scores[tw_ind] / max_coral_score
                cur_score = density_scores[tw_ind] * \
                            math.exp(model.bert_classifier.coral_density_balance / normalized_coral_score)
                input_meta[-1].append(cur_score)
        ## construct src domain labels and tgt class labels
        input_labels_np = np.array(input_labels)
        input_meta_np = np.array(input_meta)
        tgt_input_y = list()
        tgt_input_y_row = list()
        src_input_y = list()
        src_input_y_row = list()
        for tw_ind in range(tw_count):
            tgt_cur_lbl = statistics.mode(input_labels_np[:, tw_ind])
            tgt_input_y.append(tgt_cur_lbl)
            tgt_input_y_row.append([0 for _ in range(class_count)])
            tgt_input_y_row[-1][tgt_cur_lbl] = 1
            meta_weights = input_meta_np[:, tw_ind]
            if sum(meta_weights) == len(meta_weights):
                ## in case an eqaul weight is assigned to all the domains (for paper)
                src_input_y.append(len(meta_weights) + 1)
                src_input_y_row.append([1 for _ in topics])
                continue
            src_cur_lbl = meta_weights.argmax()
            src_input_y.append(src_cur_lbl)
            src_input_y_row.append([0 for _ in topics])
            src_input_y_row[-1][src_cur_lbl] = 1
        return {'tgt_combined_y': tgt_input_y,
                'tgt_combined_y_row': tgt_input_y_row,
                'tgt_specific_y_list': input_labels,
                'input_meta': input_meta,
                'src_domain_y': src_input_y,
                'src_domain_y_row': src_input_y_row}

    @staticmethod
    def __expand_data(model, data, scores, src_labeled, tgt_unlabeled):
        domain_count = len(data) - 1
        data_cp = copy.deepcopy(data)
        class_count = len(tgt_unlabeled.input_y_row[0][0])
        result = list()
        ## extract the src and tgt bundles
        src_bundle = data_cp[:domain_count]
        tgt_bundle = data_cp[-1]
        ## attach the source bundles
        result.extend(src_bundle)
        ## copy the tgt bundle multiple times and set the meta values for the sampling
        info = EDomainAdaptMine1.__score_to_lbl(model, scores, src_labeled, tgt_unlabeled)
        EDomainAdaptMine1.__print_teacher_labels(src_labeled, info['src_domain_y'], info['src_domain_y_row'],
                                                 tgt_unlabeled.tws[0].Query, model.config.output_dir)
        ## attach the tgt data for coral loss
        tgt_bundle_coral = copy.deepcopy(tgt_bundle)
        result.append(tgt_bundle_coral)
        ## attach the tgt data with the domain labels for KL loss
        tgt_bundle_kl = copy.deepcopy(tgt_bundle)
        result.append(tgt_bundle_kl)
        tgt_bundle_kl.input_y = list()
        tgt_bundle_kl.input_y_row = list()
        tgt_bundle_kl.task_list = list()
        for cur_d in range(domain_count):
            tgt_bundle_kl.input_y.append(info['tgt_specific_y_list'][cur_d])
            tgt_bundle_kl.input_y_row.append(ELib.one_hot(info['tgt_specific_y_list'][cur_d], class_count))
            tgt_bundle_kl.task_list.append('tgt_' + str(cur_d))
        tgt_bundle_kl.input_y.append(info['src_domain_y'])
        tgt_bundle_kl.input_y_row.append(info['src_domain_y_row'])
        tgt_bundle_kl.task_list.append('domain_lbl')
        return result

    @staticmethod
    def __coral_loss_cov_loss(model, src, tgt):
        vector_dim = src.size(1)
        # source covariance mat
        src_coef = 1.0 / (src.size(0) - 1.0)
        src_logits_t = src.t()
        src_diff_to_mean = src_logits_t - torch.mean(src_logits_t, dim=1, keepdim=True)
        src_diff_to_mean_t = src_diff_to_mean.t()
        src_cov = src_coef * src_diff_to_mean.matmul(src_diff_to_mean_t)
        # target covariance mat
        tgt_coef = 1.0 / (tgt.size(0) - 1.0)
        tgt_logits_t = tgt.t()
        tgt_diff_to_mean = tgt_logits_t - torch.mean(tgt_logits_t, dim=1, keepdim=True)
        tgt_diff_to_mean_t = tgt_diff_to_mean.t()
        tgt_cov = tgt_coef * tgt_diff_to_mean.matmul(tgt_diff_to_mean_t)
        # coral loss
        diff = src_cov - tgt_cov
        coral_loss = (1.0 / (4 * vector_dim ^ 2)) * torch.norm(diff * diff, p='fro')
        return coral_loss

    @staticmethod
    def __kl_schedule(cur_step, all_steps):
        p = cur_step / all_steps
        alignment_weight = (2 / (1 + math.exp(-10 * p)) - 1)
        return 1 - alignment_weight

    @staticmethod
    def __train_loss(model, pred, batch):
        ## init
        t_step_cur = model.bert_classifier.train_step
        t_steps_all = model.scheduler_overall_steps
        kl_schedule = EDomainAdaptMine1.__kl_schedule(t_step_cur, t_steps_all)
        d_count = model.bert_classifier.domain_count
        zeros_mat, zeros_vec = torch.zeros((model.config.batch_size, 2)), torch.zeros(model.config.batch_size)
        if 'task_list' in batch[d_count]:
            task_name = batch[d_count]['task_list'][0][0]
        else:
            for d_ind in range(d_count):
                if 'task_list' in batch[d_ind]:
                    task_name = batch[d_ind]['task_list'][0][0]
        ## find the current state
        if model.bert_classifier.train_state == 1:
            ## collect the src and tgt vectors
            for d_ind in range(d_count):
                if len(pred['src_pooled'][d_ind]) > 0:
                    model.bert_classifier.vectors[d_ind]['src'].extend(pred['src_pooled'][d_ind].
                                                                       detach().cpu().numpy().tolist())
                if len(pred['tgt_pooled'][d_ind]) > 0:
                    model.bert_classifier.vectors[d_ind]['tgt'].extend(pred['tgt_pooled'][d_ind].
                                                                       detach().cpu().numpy().tolist())
            return torch.tensor([0.0], requires_grad=True), task_name, zeros_mat, zeros_vec
        elif model.bert_classifier.train_state == 2:
            ## train the src classifiers
            loss = 0
            for d_ind in range(d_count):
                pred_ls = F.log_softmax(pred['src_output'][d_ind], dim=1)
                cls_loss_list = -(batch[d_ind]['y_row_0'] * pred_ls).sum(dim=1)
                loss += cls_loss_list.mean()
            return loss, task_name, zeros_mat, zeros_vec
        elif model.bert_classifier.train_state == 3:
            ## train the src classifiers and correlate with the tgt data
            cls_loss_sum = 0
            coral_loss_sum = 0
            for d_ind in range(d_count):
                src_pooled = pred['src_pooled'][d_ind]
                src_output = pred['src_output'][d_ind]
                tgt_pooled = pred['tgt_pooled'][d_ind]
                ## calculate the src classifier loss over src data
                pred_ls = F.log_softmax(src_output, dim=1)
                cls_loss_list = -(batch[d_ind]['y_row_0'] * pred_ls).sum(dim=1)
                cls_loss_sum += cls_loss_list.mean()
                ## calculate the weighted coral loss
                coral_loss = EDomainAdaptMine1.__coral_loss_cov_loss(model, src_pooled, tgt_pooled)
                coral_loss_sum += model.bert_classifier.coral_scale[d_ind] * coral_loss
            loss = cls_loss_sum + coral_loss_sum
            return loss, task_name, zeros_mat, zeros_vec
        elif model.bert_classifier.train_state == 4:
            if model.bert_classifier.stage == 1:
                b0_ind = d_count
                b2_ind = d_count + 1
                c_count = len(batch[b0_ind]['y_row_0'][0].detach().cpu().numpy().tolist())
                cls_loss_sum = 0
                coral_loss_sum_unbalanced = 0
                coral_loss_sum_balanced = 0
                kl_loss_sum = 0
                kl_loss_sum_shared = 0
                for d_ind in range(d_count):
                    src_pooled = pred['src_pooled'][d_ind]
                    src_output = pred['src_output'][d_ind]
                    tgt_pooled_unbalanced = pred['tgt_pooled_unbalanced'][d_ind]
                    # tgt_pooled_balanced = pred['tgt_pooled_balanced'][d_ind]
                    tgt_output_kl = pred['tgt_output_kl'][d_ind]
                    ## calculate the src classifier loss over src data
                    pred_ls = F.log_softmax(src_output, dim=1)
                    cls_loss_list = -(batch[d_ind]['y_row_0'] * pred_ls).sum(dim=1)
                    cls_loss_sum += cls_loss_list.mean()
                    ## calculate the weighted coral loss for the unbalanced batch
                    coral_loss = EDomainAdaptMine1.__coral_loss_cov_loss(model, src_pooled, tgt_pooled_unbalanced)
                    coral_loss_sum_unbalanced += model.bert_classifier.coral_scale[d_ind] * coral_loss
                    ## claculate kl
                    s_1 = batch[b2_ind]['y_row_' + str(d_count)][:, d_ind].detach().view(-1, 1)
                    o_1_init = batch[b2_ind]['y_row_' + str(d_ind)] + sys.float_info.epsilon
                    for d_2 in range(d_count):
                        if d_ind != d_2:
                            o_2 = F.softmax(pred['tgt_output_kl'][d_2], dim=1)
                            kl_loss_cur = s_1 * (o_1_init * torch.log2(o_1_init / o_2)).sum(dim=1).view(-1, 1)
                            kl_loss_sum += kl_loss_cur.mean()
                    ## calculate shared kl
                    o_1_current = F.softmax(tgt_output_kl.detach(), dim=1)
                    for sh_ind, shared_logit in enumerate(pred['tgt_output_shared']):
                        shared_out = F.softmax(shared_logit, dim=1)
                        kl_loss_shared = (o_1_current * torch.log2(o_1_current / shared_out)).sum(dim=1).view(-1, 1)
                        kl_loss_sum_shared += kl_loss_shared.mean()
                    ELib.PASS()
                coral_loss_sum_balanced = (coral_loss_sum_balanced / c_count)
                kl_loss_all =(kl_loss_sum/(d_count-1)) * model.bert_classifier.kl_scale * kl_schedule + kl_loss_sum_shared
                loss = cls_loss_sum + coral_loss_sum_unbalanced + coral_loss_sum_balanced + kl_loss_all
            model.bert_classifier.stage = model.bert_classifier.stage % 1 + 1
            return loss, task_name, zeros_mat, zeros_vec
        return None

    @staticmethod
    def __test_loss(model, pred, batch):
        loss = 0
        logits = None
        active_d = model.bert_classifier.active_domain
        d_count = model.bert_classifier.domain_count
        if active_d == -1:
            ## aggergating all of the classifiers
            sum_pred = 0
            cls_count = len(pred['logits'])
            scores = torch.ones(batch['x'].size(0), cls_count).to(model.config.device) / cls_count
            for ind, cur_pred in enumerate(pred['logits']):
                cur_softmax = F.softmax(cur_pred, dim=1)
                cur_softmax_weighted = scores[:, ind].view(-1, 1) * cur_softmax
                sum_pred += cur_softmax_weighted
            loss = -(batch['y_row_0'] * torch.log(sum_pred)).sum(dim=1)
            loss = loss.mean()
            logits = sum_pred
        elif active_d < d_count:
            ## evaluating a specific src domain
            logits = pred['logits'][active_d]
            loss = -(batch['y_row_0'] * torch.log_softmax(logits, dim=1)).sum(dim=1)
            loss = loss.mean()
        return loss, batch['task_list'][0][0], logits, batch['y_0']

    @staticmethod
    def run(cur_itr, lc, model_path, output_dir, device, device_2, seed, src_labeled, src_unlabeled,
            tgt_labeled, tgt_unlabeled, param):
        coral_scales = [1.0, 0.1, 0.01, 0.001, 0.0001]
        ## copy the data and create a list of bundles for the training
        domain_count = len(src_labeled)
        data = copy.deepcopy(src_labeled)
        EDomainAdaptMine1.__replicate_short_sets(data, lc, seed)
        data.append(copy.deepcopy(tgt_unlabeled))
        ## setup the model
        cls = EDomainAdaptMine1.__get_cls(model_path, output_dir, device, device_2, seed, src_labeled, tgt_unlabeled)
        ## preparing the scores
        scores = EDomainAdaptMine1.__scores(cls, data, src_labeled, tgt_unlabeled, coral_scales)
        ## training all the modules
        print(colored('training the experts...', 'green'))
        cls.config.epoch_count = 3
        cls.bert_classifier.coral_density_balance = 1.0
        cls.bert_classifier.kl_scale = 0.9 ## change this to experiment with KL (default is 0.9)
        cls.bert_classifier.train_state = 4
        cls.bert_classifier.train_step = -1
        sc, en = EDomainAdaptMine1.__get_scales_and_encoders(lc, src_labeled, tgt_unlabeled, scores, coral_scales)
        cls.bert_classifier.reconfig_encoders(sc, en)
        data_exp = EDomainAdaptMine1.__expand_data(cls, data, scores, src_labeled, tgt_unlabeled)
        data_mode_exp = [EBalanceBatchMode.label_based for _ in range(domain_count)] + \
                        [EBalanceBatchMode.none, EBalanceBatchMode.none]
        cls.train(data_exp, input_mode=EInputListMode.parallel, balance_batch_mode_list=data_mode_exp)
        ## label
        print(colored('labeling...', 'green'))
        EDomainAdaptMine1.__domain_test(cls, src_labeled, tgt_labeled)
        cls.bert_classifier.active_domain = -1
        _, _, _, perf = cls.test(tgt_labeled)
        del cls
        gc.collect()
        return perf

